"""Opt-in M-stage metadata/policy bridge. NEVER authorizes a physical HIT.

Actual distributed sparse projection execution is intentionally fail-closed:
R5's versioned correction protocol has no gather/scatter token-position contract.
"""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace
import time
from semcache.semantic.subsequence import SubsequenceExtractor
from semcache.semantic.matcher import ExactTokenMatcher
from semcache.semantic.hit_selection import CacheHit,select_nonoverlapping
from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.cache.cache_metrics import normalize
from semcache.cache.metric_manager import CacheMetricManager,actual_attention_impact
from .communication import require,digest
from .r5_i5_contracts import validate_request
from .r5_multi_artifacts import resolve,sharing

MODE='MSTAGE_METADATA_SHADOW_ONLY'
BLOCKERS=('SPARSE_POSITION_TCP_CONTRACT_NOT_IMPLEMENTED','DISTRIBUTED_TOTAL_QKV_MATERIALIZATION_AND_REPLAY_NOT_IMPLEMENTED',
          'CONTEXT_DEPENDENT_HIDDEN_STATE_EQUIVALENCE_UNPROVEN','ACTUAL_ATTENTION_IMPACT_CAPTURE_NOT_CONNECTED',
          'INDEPENDENT_SUBSEQUENCE_TENSOR_AUDIT_NOT_IMPLEMENTED')

class BoundedCache(GlobalCache):
    """Use unchanged M scores; bound entries as well as stored logical bytes."""
    def insert(self,*args,on_event=None,**kwargs):
        result=super().insert(*args,on_event=on_event,**kwargs)
        while len(self.entries)>64:
            ms={k:self._metrics(e) for k,e in self.entries.items()}
            key=min(ms,key=lambda k:(-self.eviction.score(normalize(ms[k],ms.values(),self.zero_max)),k))
            if on_event:on_event('EVICT',self.entries[key],None)
            del self.entries[key]
        return result and args[0].key in self.entries

class MStageShadow:
    """One physical model/GPU/adapter namespace. Logical tensors are never HITs.

    Assignment comes from the actual R5 TinyBERT/cluster result; no fabricated
    embeddings or duplicate semantic execution. Tests explicitly use fixtures.
    """
    def __init__(self,identity,registry,*,capacity=512*2**20,max_candidates=64,pbr_interval=100,trace=lambda e:None):
        require(0<capacity<=512*2**20 and 1<=max_candidates<=64 and pbr_interval>=1,'bounded shadow configuration required')
        self.identity=deepcopy(identity);self.registry=registry;self.cache=BoundedCache(capacity)
        self.metrics=CacheMetricManager(self.cache);self.extractor=SubsequenceExtractor(3);self.matcher=ExactTokenMatcher()
        self.max_candidates=max_candidates;self.pbr_interval=pbr_interval;self.trace=trace
        self.pending=None;self.seen=set();self.physical_hits=0;self.skipped_rows=0
    def emit(self,kind,**fields):
        self.trace(dict(kind=kind,mode=MODE,physical_hit=False,skipped_projection_rows=0,skipped_wire_bytes=0,**fields))
    def begin(self,req,assignment):
        validate_request(req)
        require(req['physical_identity']==self.identity and resolve(self.registry,req['user_id'],req['adapter_id'])==self.identity['adapter_id'],'physical/adapter identity differs')
        require(self.pending is None and req['request_id'] not in self.seen,'pending or duplicate request')
        require(type(assignment.get('cluster_id')) is int and assignment['cluster_id']>=0,'actual cluster assignment required')
        self.seen.add(req['request_id']);start=time.monotonic_ns()
        cluster=assignment['cluster_id'];windows=self.extractor.extract(req['token_ids'])
        self.metrics.arrive([self.matcher.key(cluster,w) for w in windows])
        hits=[];candidates=[]
        for window in windows:
            entry=self.cache.lookup(self.matcher.key(cluster,window),record_reuse=False)
            if entry is None:continue
            meta=entry.qkv_metadata
            reasons=[]
            denied=sharing(self.registry,meta['source_user'],req['user_id'],meta['source_adapter'],req['adapter_id'])
            if denied:reasons.append(denied)
            if meta['physical_identity']!=req['physical_identity']:reasons.append('PHYSICAL_IDENTITY_MISMATCH')
            if entry.positions!=(window.start,window.end):reasons.append('SHIFTED_POSITION')
            if meta['source_token_ids'][:window.end]!=req['token_ids'][:window.end]:reasons.append('CAUSAL_PREFIX_DIFFERS')
            # These are diagnostic checks, not sufficient numerical proofs.
            candidates.append(dict(source_request_id=meta['source_request_id'],source_user=meta['source_user'],source_adapter=meta['source_adapter'],source_positions=list(entry.positions),target_positions=[window.start,window.end],token_ids=list(window.token_ids),rejections=reasons,
                                   physical_rejection=list(BLOCKERS),component_scope='metadata_only'))
            if not denied and meta['physical_identity']==req['physical_identity']:hits.append(CacheHit(window,entry,entry.impact or 0.))
            if len(candidates)>=self.max_candidates:break
        selected,mask=select_nonoverlapping(hits,len(req['token_ids']))
        self.pending=dict(req=deepcopy(req),cluster=cluster,windows=windows,selected=selected)
        report=dict(request_id=req['request_id'],candidates=candidates,selected_spans=[[h.window.start,h.window.end] for h in selected],candidate_rows=sum(mask),
                    selection_scope='SHADOW_ONLY; shifted/causally incompatible spans may be diagnostic candidates, never physical grants',candidate_limit=self.max_candidates,
                    chu_status='NOT_APPLIED_NO_PHYSICAL_REUSE',policy_lookup_ms=(time.monotonic_ns()-start)/1e6)
        self.emit('MSTAGE_MATCH',**report);return report
    def complete(self,*,completed,attentions=None,valid_attention_mask=None):
        require(self.pending is not None and completed is True,'completed standalone source required')
        p=self.pending;start=time.monotonic_ns();req=p['req'];cluster=p['cluster']
        if attentions is None:
            self.emit('MSTAGE_ADMISSION_UNAVAILABLE',request_id=req['request_id'],reason='ACTUAL_ATTENTION_PROBABILITIES_NOT_RECORDED');self.pending=None;return []
        # No caller-supplied scalar importance surrogate is accepted. Use the
        # actual M-stage reducer, including valid-mask and finite-value checks.
        values={}
        for w in p['windows']:
            key=self.matcher.key(cluster,w);values.setdefault(key,[]).append(actual_attention_impact(attentions,w.start,w.end,valid_attention_mask))
        self.metrics.history.append(cluster,req['request_id'],self.cache.now,{k:sum(v)/len(v) for k,v in values.items()})
        outcomes=[];considered=set()
        for w in p['windows']:
            key=self.matcher.key(cluster,w)
            if key in considered or key in self.cache.entries:continue
            considered.add(key)
            size=3*len(w.token_ids)*self.identity['layers']*self.identity['hidden_size']*(2 if self.identity['dtype']=='float16' else 4)
            entry=CacheEntry(cluster,w.token_ids,(w.start,w.end),size,impact=values[key][0],qkv_metadata=dict(component_scope='metadata_only',physical_identity=deepcopy(self.identity),source_user=req['user_id'],source_adapter=req['adapter_id'],source_request_id=req['request_id'],source_token_ids=list(req['token_ids']),completed=True))
            admitted=self.cache.insert(entry,observed_frequency=self.metrics.frequencies[key],admission_frequencies=self.metrics.frequencies,
                                       on_event=lambda kind,e,score:self.emit('MSTAGE_'+kind,key=e.key,source_request_id=e.qkv_metadata['source_request_id'],score=score))
            outcomes.append(dict(key=key,admitted=admitted,F=entry.frequency,A=entry.age(self.cache.now),I=entry.impact,S=size))
        if self.cache.now%self.pbr_interval==0:self.emit('MSTAGE_PBR',updates=self.metrics.pbr(cluster))
        self.emit('MSTAGE_COMPLETE',request_id=req['request_id'],outcomes=outcomes,logical_bytes=self.cache.logical_cache_bytes,physical_tensor_bytes=0,impact_admission_policy_ms=(time.monotonic_ns()-start)/1e6)
        self.pending=None;return outcomes
    def chu_preview(self,entry,current_actual_impact):
        """M formula inspection only: never increment actual-use F on lookup."""
        preview=replace(entry,qkv_metadata=deepcopy(entry.qkv_metadata))
        self.metrics.updater.on_hit(preview,current_actual_impact);return preview.impact
    def physical_grant(self,*args,**kwargs):
        raise NotImplementedError('SEMCACHE_SUBSEQUENCE is not executable: '+', '.join(BLOCKERS))
    def dispose(self):
        self.cache.entries.clear();self.pending=None

@contextmanager
def shadow_controller_factory(runtime,*,trace=lambda e:None):
    """Explicit process-local R5 integration; restore original factory in finally.

    FULL_RECOMPUTE and existing SEMCACHE_INTEGRATED execute their original paths.
    B additionally observes real semantic assignments. R5 does not supply actual
    attention probabilities, so no fake impact or admitted tensor is generated.
    """
    original=runtime.I5Controller
    class ShadowController(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.mstage=MStageShadow(self.identity,self.registry,trace=trace) if self.condition=='SEMANTIC_NO_REUSE' else None
        def decide(self,req):
            event,found=super().decide(req)
            if self.mstage and not req['request_id'].startswith('W_'):self.mstage.begin(req,event['assignment'])
            return event,found
        def admit(self,req,assignment,blocks,proof,*,completed):
            grant=super().admit(req,assignment,blocks,proof,completed=completed)
            if self.mstage and not req['request_id'].startswith('W_'):self.mstage.complete(completed=completed)
            return grant
    runtime.I5Controller=ShadowController
    try:yield
    finally:runtime.I5Controller=original
