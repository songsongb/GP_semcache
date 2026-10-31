"""A: zero semantic calls. B: metadata-only virtual LRU. C: physical I4A reuse."""
from collections import OrderedDict
from copy import deepcopy
import time
from .communication import canonical, digest, require
from .r5_multi_artifacts import resolve
from .r5_multi_controller import reason
from .r5_i5_contracts import validate_request, prompt
from .r5_i5_workload import CONDITIONS


class DisabledIndex:
    entries = {}
    def invalidate(self, key): pass
    def state(self): return dict(disabled=True, queries=0)


class I5Controller:
    def __init__(self, owner, index, identity, *, condition, registry):
        require(condition in CONDITIONS, 'unsupported I5 condition')
        self.owner=owner;self.index=index;self.identity=identity;self.condition=condition;self.registry=registry
        self.reuse=condition=='SEMCACHE_INTEGRATED';self.seen=set();self.shadow=OrderedDict();self.shadow_peak=0
        self.virtual_selected=None

    def decide(self, req):
        validate_request(req)
        require(resolve(self.registry,req['user_id'],req['adapter_id'])==self.identity['adapter_id'] and req['physical_identity']==self.identity, 'UNAUTHORIZED_USER_OR_ADAPTER')
        require(req['request_id'] not in self.seen, 'duplicate request');self.seen.add(req['request_id'])
        self.owner.request_id=req['request_id'];self.owner.active_request=deepcopy(req);self.virtual_selected=None
        if self.condition=='FULL_RECOMPUTE':
            return dict(assignment=dict(cluster_id=None,encoder=None,semantic_encoding_ms=0.,cluster_assignment_ms=0.),
                candidates=[],candidate_reasons=[],selected_entry_id=None,outcome='CACHE_MISS',
                candidate_lookup_ms=0.,eligibility_validation_ms=0.,cache_lookup_and_kv_ms=0.,semantic_disabled=True),None
        assignment=self.index.assign(req['input_text']);candidate_start=time.monotonic_ns()
        candidates=self.index.discover(assignment['cluster_id'],req['token_ids']);looked=time.monotonic_ns()
        resident=self.owner.entries if self.reuse else self.shadow
        reasons=[dict(entry_id=c['metadata']['entry_id'],source_request_id=c['metadata']['source_request_id'],
            prefix_length=len(c['metadata']['token_ids']),rejection=reason(c['metadata'],req,resident,self.registry)) for c in candidates]
        eligible=[r for r in reasons if r['rejection'] is None];validated=time.monotonic_ns();found=None;lookup=0.
        if eligible:
            key=eligible[0]['entry_id'];self.virtual_selected=key
            if self.reuse:
                tick=time.monotonic_ns();found=self.owner.selected(key,prompt(req),req['request_id']);lookup=(time.monotonic_ns()-tick)/1e6
            else:self.shadow.move_to_end(key)
        event=dict(assignment=assignment,candidates=candidates,candidate_reasons=reasons,
            selected_entry_id=None if found is None else self.virtual_selected,virtual_selected_entry_id=self.virtual_selected,
            outcome='PHYSICAL_PREFIX_HIT' if found else 'CACHE_MISS', semantic_disabled=False,
            candidate_lookup_ms=(looked-candidate_start)/1e6,
            eligibility_validation_ms=(validated-looked)/1e6,cache_lookup_and_kv_ms=lookup)
        self.index.trace(dict(event,kind='DISCOVERY',request_id=req['request_id']))
        return event,found

    def metadata(self, req, assignment, key, proof_sha):
        return dict(entry_id=key,source_request_id=req['request_id'],source_user=req['user_id'],token_ids=list(req['token_ids']),
            positions=list(req['positions']),attention_mask=list(req['attention_mask']),physical_identity=deepcopy(req['physical_identity']),
            cluster_id=assignment['cluster_id'],encoder=deepcopy(assignment['encoder']),completed=True,
            request_sha256=req['request_sha256'],proof_sha256=proof_sha)

    def admit(self, req, assignment, blocks, proof, *, completed):
        require(completed is True, 'uncompleted source')
        if self.condition=='FULL_RECOMPUTE':return None
        # Match C: a HIT has no complete standalone QKV and is not admitted.
        if self.virtual_selected is not None:return None
        size=len(req['token_ids'])*3*self.identity['layers']*self.identity['hidden_size']*(4 if req['TEST_FIXTURE_ONLY'] else 2)
        if size>self.owner.capacity_bytes or len(req['token_ids'])>=2048:
            self.owner.event('ADMISSION_DENIED',reason='CAPACITY_OR_CONTEXT_LIMIT');return None
        if self.reuse:
            require(blocks is not None and proof is not None, 'missing physical standalone tensors')
            grant=self.owner.admit(prompt(req),blocks,proof,req['request_id'],completed=True)
            if grant:self.index.add(grant['entry_id'],self.metadata(req,assignment,grant['entry_id'],grant['proof_sha256']))
            return grant
        key=digest(self.owner.context_for(prompt(req)))
        def remove(k):
            self.shadow.pop(k);self.index.invalidate(k);self.owner.event('SHADOW_EVICT',context_sha256=k)
        if key in self.shadow:remove(key)
        while len(self.shadow)>=64 or sum(self.shadow.values())+size>self.owner.capacity_bytes:remove(next(iter(self.shadow)))
        self.shadow[key]=size;self.index.add(key,self.metadata(req,assignment,key,'SHADOW_NO_TENSORS'))
        self.shadow_peak=max(self.shadow_peak,sum(self.shadow.values()))
        self.owner.event('SHADOW_ADMISSION',context_sha256=key,simulated_bytes=size)
        return None

    def shadow_state(self):
        return dict(simulated_raw_capacity_bytes=sum(self.shadow.values()),simulated_peak_bytes=self.shadow_peak,
            metadata_serialized_bytes=len(canonical(dict(entries=self.index.entries,postings={str(k):sorted(v) for k,v in getattr(self.index,'postings',{}).items()}))),
            python_object_heap_bytes='NOT_MEASURED',gpu_tensor_bytes=0,entries=len(self.shadow))
