"""Automatic routing over the authoritative R4-C1 sealed Raw-QKV owner."""
from copy import deepcopy
import time

from .communication import digest, require
from .r3b_cache import spec
from .r4b2_kv import attention_kv, construct_past
from .r4c_cache import PrefixCache
from .r5_contracts import prompt, validate_request
from .r5_semantic_index import eligibility


class CacheIntegrityError(ValueError):
    pass


class RawOwner(PrefixCache):
    def __init__(self,*args,index,**kwargs):
        self.index=index
        super().__init__(*args,**kwargs)

    def remove(self,key,kind,reason):
        self.index.invalidate(key)
        super().remove(key,kind,reason)

    def selected(self,key,target,request_id):
        """Consume only the controller-selected candidate; never scan around filtering."""
        started=time.monotonic_ns();self.request_id=request_id
        try:
            self.check_namespace();self.context_for(target)
            owner=self.entries[key];c=owner.context;n=len(c['token_ids'])
            require(c==self.context_for(dict(token_ids=target['token_ids'][:n],attention_mask=[1]*n)), 'stored prefix context mismatch')
            require(n<len(target['token_ids']) and c['token_ids']==target['token_ids'][:n], 'invalid exact prefix')
            require(digest(self.proofs[key])==self.proof_seals[key], 'corrupted prefix proof digest')
            entry=next(iter(owner.pool.entries.values()));owner.validate(entry,c)
            proof=self.proofs[key];validated=time.monotonic_ns()
            pairs=attention_kv(entry.tensors,proof['heads'],self._pm['identity']['hidden_size'])
            for i,pair in enumerate(pairs):
                require(spec(pair[0])==proof['layers'][i]['actual_k'] and spec(pair[1])==proof['layers'][i]['actual_v'], 'derived KV integrity failed')
            past=construct_past(pairs,self.cache_type,proof['cache_api']);self.synchronize()
        except (ValueError,KeyError,TypeError,AttributeError,IndexError) as exc:
            self.remove(key,'REJECT','INVALID_CACHE_INTEGRITY: '+str(exc))
            self.event('INVALID_CACHE_INTEGRITY',context_sha256=key,reason=str(exc))
            raise CacheIntegrityError('INVALID_CACHE_INTEGRITY: '+str(exc)) from exc
        self.last_lookup_timings=dict(cache_lookup_validation_ms=(validated-started)/1e6,
            kv_construction_and_validation_ms=(time.monotonic_ns()-validated)/1e6)
        # LRU changes only after complete physical and identity validation.
        self.entries.move_to_end(key)
        self.event('HIT',context_sha256=key,source_request_id=entry.qkv_metadata['source_request_id'],validation_read_bytes=entry.size_bytes)
        grant=dict(entry_id=key,source_request_id=entry.qkv_metadata['source_request_id'],prefix_length=n,proof_sha256=self.proof_seals[key])
        return entry,grant,past


class LRUPolicy:
    name='LRU_FIT_COMPLETE_SOURCE_V1'
    def eligible(self,request,blocks):
        if blocks is None:return 'NO_COMPLETE_STANDALONE_RAW_QKV'
        if len(request['token_ids'])>=512:return 'SOURCE_LENGTH_NOT_REUSABLE'
        return None


class Controller:
    def __init__(self,owner,index,physical_identity,*,reuse=True,policy=None):
        self.owner=owner;self.index=index;self.identity=deepcopy(physical_identity)
        self.reuse=reuse;self.policy=policy or LRUPolicy();self.seen=set()

    def decide(self,request):
        validate_request(request)
        require(request['request_id'] not in self.seen, 'duplicate request ID')
        require(request['physical_identity']==self.identity,'UNSUPPORTED_ADAPTER_OR_MODEL: request namespace unprepared')
        self.seen.add(request['request_id']);self.owner.request_id=request['request_id']
        assignment=self.index.assign(request['input_text']);tick=time.monotonic_ns()
        candidates=self.index.discover(assignment['cluster_id'],request['token_ids']);looked=time.monotonic_ns()
        reasons=[];eligible=[]
        for candidate in candidates:
            m=candidate['metadata'];reason=eligibility(m,request,self.owner.entries)
            reasons.append(dict(entry_id=m['entry_id'],source_request_id=m['source_request_id'],rejection=reason,
                prefix_length=len(m['token_ids'])))
            if reason is None:eligible.append(m)
        chosen=eligible[0] if eligible and self.reuse else None
        validation_ms=(time.monotonic_ns()-looked)/1e6;found=None;lookup=0.
        if chosen is not None:
            start=time.monotonic_ns();found=self.owner.selected(chosen['entry_id'],prompt(request),request['request_id']);lookup=(time.monotonic_ns()-start)/1e6
        outcome='PHYSICAL_PREFIX_HIT' if found else 'SEMANTIC_CANDIDATE_REJECTED' if candidates and not eligible else 'CACHE_MISS'
        if found is None:self.owner.event('MISS',context_sha256=None,reason=outcome if self.reuse else 'DISTRIBUTED_NO_REUSE')
        event=dict(kind='DISCOVERY',request_id=request['request_id'],assignment=assignment,candidates=candidates,
            candidate_reasons=reasons,selected_entry_id=None if chosen is None else chosen['entry_id'],outcome=outcome,
            physical_reuse_enabled=self.reuse,candidate_lookup_ms=(looked-tick)/1e6,eligibility_validation_ms=validation_ms,
            cache_lookup_and_kv_ms=lookup,index_entry_ids=sorted(self.index.entries))
        self.index.trace(event)
        return event,found

    def admit(self,request,assignment,blocks,proof,*,completed):
        require(completed is True,'source generation incomplete')
        self.owner.request_id=request['request_id']
        reason='NO_REUSE_BASELINE' if not self.reuse else self.policy.eligible(request,blocks)
        if reason:
            self.owner.event('ADMISSION_DENIED',context_sha256=None,reason=reason);return None
        grant=self.owner.admit(prompt(request),blocks,proof,request['request_id'],completed=True)
        if grant is None:return None
        metadata=dict(entry_id=grant['entry_id'],source_request_id=request['request_id'],source_user=request['user_id'],
            token_ids=list(request['token_ids']),positions=list(request['positions']),attention_mask=list(request['attention_mask']),
            physical_identity=deepcopy(request['physical_identity']),cluster_id=assignment['cluster_id'],
            encoder=deepcopy(assignment['encoder']),completed=True,request_sha256=request['request_sha256'],
            proof_sha256=grant['proof_sha256'],admitted_monotonic_ns=time.monotonic_ns())
        self.index.add(grant['entry_id'],metadata)
        return grant
