"""Private adapter caches with explicit registry-authorized sharing."""
from copy import deepcopy
import time

from .communication import digest,require
from .r5_contracts import prompt,validate_request
from .r5_controller import Controller,RawOwner
from .r5_semantic_index import eligibility
from .r5_multi_artifacts import resolve,sharing,validate_registry


def reason(metadata,request,resident,registry):
    denied=sharing(registry,metadata['source_user'],request['user_id'],metadata['physical_identity']['adapter_id'],request['adapter_id'])
    return denied or eligibility(metadata,request,resident)


class MultiOwner(RawOwner):
    def __init__(self,*args,spec,registry,**kwargs):
        self.adapter_spec=deepcopy(spec);self.registry=deepcopy(registry);self.active_request=None
        super().__init__(*args,**kwargs)
    def context_for(self,prompt_value):
        context=super().context_for(prompt_value)
        context['r5_multi']=dict(schema='R5_MULTI_CACHE_NAMESPACE_V1',adapter_spec_sha256=digest(self.adapter_spec),
            registry_sha256=self.registry['sha256'],sharing_policy=self.registry['policy'],scale=self.adapter_spec['scale'],
            layout=self.adapter_spec['layout'],cache_format='UNCHANGED_R3B_TOTAL_RAW_QKV')
        return context
    def selected(self,key,target,request_id):
        req=self.active_request
        require(req is not None and req['request_id']==request_id and req['adapter_id']==self.adapter_spec['adapter_id'] and
                resolve(self.registry,req['user_id'],req['adapter_id'])==req['adapter_id'],'UNAUTHORIZED_CACHE_NAMESPACE')
        require(key in self.index.entries,'FOREIGN_OR_STALE_CACHE_ENTRY')
        require(reason(self.index.entries[key],req,self.entries,self.registry) is None,'UNAUTHORIZED_CACHE_SHARING')
        return super().selected(key,target,request_id)


class MultiController(Controller):
    def __init__(self,*args,registry,**kwargs):
        self.registry=deepcopy(registry);super().__init__(*args,**kwargs)
    def decide(self,request):
        validate_request(request)
        require(resolve(self.registry,request['user_id'],request['adapter_id'])==self.identity['adapter_id'] and
                request['physical_identity']==self.identity,'UNAUTHORIZED_USER_OR_ADAPTER')
        require(request['request_id'] not in self.seen,'duplicate request ID')
        self.seen.add(request['request_id']);self.owner.request_id=request['request_id'];self.owner.active_request=deepcopy(request)
        assignment=self.index.assign(request['input_text']);tick=time.monotonic_ns()
        candidates=self.index.discover(assignment['cluster_id'],request['token_ids']);looked=time.monotonic_ns()
        reasons=[dict(entry_id=c['metadata']['entry_id'],source_request_id=c['metadata']['source_request_id'],
            prefix_length=len(c['metadata']['token_ids']),rejection=reason(c['metadata'],request,self.owner.entries,self.registry)) for c in candidates]
        eligible=[r['entry_id'] for r in reasons if r['rejection'] is None]
        chosen=eligible[0] if eligible and self.reuse else None;validated=time.monotonic_ns();found=None;lookup=0.
        if chosen is not None:
            start=time.monotonic_ns();found=self.owner.selected(chosen,prompt(request),request['request_id']);lookup=(time.monotonic_ns()-start)/1e6
        outcome='PHYSICAL_PREFIX_HIT' if found else 'SEMANTIC_CANDIDATE_REJECTED' if candidates and not eligible else 'CACHE_MISS'
        if found is None:self.owner.event('MISS',context_sha256=None,reason=outcome if self.reuse else 'DISTRIBUTED_NO_REUSE')
        event=dict(kind='DISCOVERY',request_id=request['request_id'],assignment=assignment,candidates=candidates,candidate_reasons=reasons,
            selected_entry_id=chosen,outcome=outcome,physical_reuse_enabled=self.reuse,candidate_lookup_ms=(looked-tick)/1e6,
            eligibility_validation_ms=(validated-looked)/1e6,cache_lookup_and_kv_ms=lookup,index_entry_ids=sorted(self.index.entries))
        self.index.trace(event);return event,found


def reject_foreign(request,other_adapters,registry):
    """Privileged isolation probe; only aggregate reasons enter the audit.

    No foreign content, entry ID, source user, or candidate metadata is returned
    to a logical user. No foreign tensor is looked up or reconstructed.
    """
    rows=[]
    for aid,state in sorted(other_adapters.items()):
        if aid==request['adapter_id']:continue
        index=state['index'];candidates=index.discover(0,request['token_ids'])
        for c in candidates:
            m=c['metadata'];denied=sharing(registry,m['source_user'],request['user_id'],aid,request['adapter_id'])
            require(denied=='CROSS_ADAPTER_REUSE_FORBIDDEN','cross-adapter candidate unexpectedly authorized')
        if candidates:rows.append(dict(foreign_adapter_id=aid,rejected_count=len(candidates),reason='CROSS_ADAPTER_REUSE_FORBIDDEN'))
    return rows


def public_result(row):
    return {key:deepcopy(row[key]) for key in ('request_id','status','generated_token_ids','generated_text','termination')}
