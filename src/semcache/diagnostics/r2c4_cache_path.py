"""Scoped cache-path oracles and exact request-boundary observations.

Single-threaded diagnostic only. Existing descriptor/cache/executor code stays
unchanged and is always fully checked outside the timer around an oracle use.
"""
from contextlib import contextmanager, ExitStack
from collections import Counter,defaultdict
from time import perf_counter_ns
from unittest.mock import patch
from semcache.diagnostics.r1b3_break_even import MissingEvidence,ParityFailure
from semcache.diagnostics.r1b3_fast_selection import tensor_stamp
from semcache.diagnostics.r1b3_gemm_shape import tensor_sha
from semcache.metrics.r1 import digest

ARMS=('FULL_NATIVE','SEMCACHE_NO_REUSE_IMPACT_REPLAY','RAW_REPLAY_BASELINE',
      'RAW_REPLAY_SNAPSHOT_ORACLE','RAW_REPLAY_MAPPING_ORACLE','RAW_REPLAY_BOTH_ORACLE')
RAW=ARMS[2:]
SNAPSHOT=(ARMS[3],ARMS[5]); MAPPING=(ARMS[4],ARMS[5])
UNSET=object()


def hit_sequence(hits):
    return tuple((id(h.entry),h.entry.qkv_metadata['source_query_id'],tuple(h.entry.positions),
        h.window.start,h.window.end,tuple(h.window.token_ids),h.entry.qkv_metadata['occurrence_identity']) for h in hits)


def source_content(runtime):
    if runtime.canonical_hashes()!=runtime.source_hashes:
        raise ParityFailure('canonical Source QKV content mutated')


def plan_state(plan,content=True):
    """Local seal: preserve owner pointers/offsets/strides and actual index bytes.

    cache_hits is intentionally excluded; it is lookup accounting, not geometry.
    No source QKV cloning, packing or replacement is performed.
    """
    descriptor=plan.descriptor; catalog=descriptor.catalog
    groups=[]
    for g in plan.groups:
        owner=catalog.sources[g['query']]['qkv']
        if g['backing'] is not owner: raise MissingEvidence('mapping Source backing ownership changed')
        groups.append((g['query'],id(owner),tuple(g['source']),tuple(g['target']),g['is_contiguous'],
            tuple((layer,tuple((id(t),tensor_stamp(t)) for t in qkv)) for layer,qkv in owner.items())))
    return (id(plan),id(descriptor),descriptor.snapshot_id,plan.adapter,plan.namespace,plan.weights,
        plan.device,plan.dtype,plan.length,tuple(plan.target_ids),tuple(plan.selected_sequence),
        bytes(plan.mask),tuple(plan.fresh),plan.reused,tuple(groups),tuple(sorted(plan.prepared_modes)),
        tuple((k,id(t),tensor_stamp(t),tensor_sha(t) if content else None) for k,t in sorted(plan.index_tensors.items())))


def validate_plan(runtime,plan,hits):
    """Outside-clock full geometry and real index contents, including warm plans."""
    import torch
    ids=tuple(runtime.ids); n=len(ids)
    if (plan.descriptor is not runtime.descriptor or plan.catalog is not runtime.catalog or plan.index is not runtime.index or
        plan.target_ids!=ids or plan.length!=n or plan.adapter!='user_a' or plan.namespace!=runtime.namespace or
        plan.weights!=runtime.weights or plan.device!=str(runtime.a.device) or plan.dtype!=str(getattr(torch,runtime.a.dtype))):
        raise MissingEvidence('prebound mapping target/model/adapter/device/dtype identity invalid')
    selected=tuple((h.entry.qkv_metadata['source_query_id'],*h.entry.positions,h.window.start,h.window.end,h.entry.token_ids,id(h.entry)) for h in hits)
    if plan.selected_sequence!=selected:raise MissingEvidence('prebound selected occurrence order changed')
    mask=[0]*n; expected_groups={}
    for h in hits:
        e,w=h.entry,h.window;q=e.qkv_metadata['source_query_id'];s,t=e.positions;source=runtime.catalog.sources[q]
        if (runtime.catalog.entries.get(e.qkv_metadata['occurrence_identity']) is not e or (s,t)!=(w.start,w.end) or
            not 0<=s<t<=n or t-s!=3 or any(mask[s:t]) or ids[s:t]!=tuple(w.token_ids) or e.token_ids!=ids[s:t] or
            source['ids'][:t]!=ids[:t] or source['adapter']!='user_a'):
            raise MissingEvidence('prebound Source/window/prefix/provenance geometry invalid')
        mask[s:t]=[1]*3;g=expected_groups.setdefault(q,[[],[]]);g[0].extend(range(s,t));g[1].extend(range(w.start,w.end))
    fresh=tuple(i for i,v in enumerate(mask) if not v)
    if plan.mask!=bytes(mask) or plan.fresh!=fresh or plan.reused!=sum(mask):raise MissingEvidence('prebound mask/fresh/skip geometry changed')
    if tuple(g['query'] for g in plan.groups)!=tuple(expected_groups) or 'contiguous' not in plan.prepared_modes:
        raise MissingEvidence('prebound grouping/prepared mode changed')
    expected_indices={('fresh',0):(fresh,runtime.a.device)}
    for number,g in enumerate(plan.groups):
        src,dst=map(tuple,expected_groups[g['query']]);owner=runtime.catalog.sources[g['query']]['qkv']
        contiguous=bool(src and all(b==a+1 for a,b in zip(src,src[1:])) and all(b==a+1 for a,b in zip(dst,dst[1:])))
        if g['backing'] is not owner or g['source']!=src or g['target']!=dst or g['is_contiguous']!=contiguous:
            raise MissingEvidence('prebound owner/offset/stride/fallback grouping changed')
        expected_indices[('source',number)]=(src,next(iter(owner.values()))[0].device)
        expected_indices[('target',number)]=(dst,runtime.a.device)
        if not contiguous and any((role,number) not in plan.index_tensors for role in ('source','target')):
            raise MissingEvidence('fallback mapping indices absent')
    if ('fresh',0) not in plan.index_tensors:raise MissingEvidence('fresh native projection indices absent')
    def normalized(device):
        d=torch.device(device)
        return torch.device(f'cuda:{torch.cuda.current_device()}') if d.type=='cuda' and d.index is None else d
    for key,tensor in plan.index_tensors.items():
        if key not in expected_indices:raise MissingEvidence('unexpected prepared indices')
        values,device=expected_indices[key]
        if (tensor.dtype!=torch.long or tuple(tensor.shape)!=(len(values),) or normalized(tensor.device)!=normalized(device) or
            tuple(tensor.detach().cpu().tolist())!=values):raise MissingEvidence('prepared index content/shape/dtype/device changed')


class OracleLease:
    def __init__(self,runtime,arm):
        self.r=runtime; self.arm=arm; self.costs={}; self.plan=None; self.plan_seal=None
        self.binding=runtime.binding('RAW_SEMCACHE_QKV')
        self.descriptor=runtime.descriptor; self.cache=runtime.caches['contiguous']
        if self.descriptor.catalog is not runtime.catalog or self.descriptor.index is not runtime.index or self.cache.descriptor is not self.descriptor:
            raise MissingEvidence('oracle descriptor/index/cache Source ownership mismatch')
        self.identity=self.identity_now()
        self.hits=tuple(self.measure('preselection_ms',lambda:runtime.selected_hits(runtime.coverage)))
        self.sequence=hit_sequence(self.hits); self.applicable=bool(self.hits)
        self.snapshot_oracle=arm in SNAPSHOT and self.applicable
        self.mapping_oracle=arm in MAPPING and self.applicable
        self.prebind_cache_hit=None
        self.measure('outside_pre_guard_ms',runtime.guard)
        self.measure('outside_pre_source_content_ms',lambda:source_content(runtime))
        if self.applicable:
            self.measure('outside_full_snapshot_validation_ms',lambda:self.descriptor.validate(self.hits))
        if self.mapping_oracle:
            import torch
            def bind():
                return self.cache.get(self.hits,runtime.ids,'user_a',runtime.namespace,runtime.weights,
                    runtime.a.device,getattr(torch,runtime.a.dtype),mode='contiguous',validate=False)
            self.plan,self.prebind_cache_hit=self.measure('outside_mapping_prebind_ms',bind)
            self.measure('outside_mapping_geometry_ms',lambda:validate_plan(runtime,self.plan,self.hits))
            self.measure('outside_index_ready_sync_ms',lambda:__import__('semcache.metrics.r1_runtime',fromlist=['synchronize']).synchronize(runtime.a.device))
            self.plan_seal=self.measure('outside_mapping_seal_ms',lambda:plan_state(self.plan))
    def measure(self,name,fn):
        start=perf_counter_ns()
        try: return fn()
        finally: self.costs[name]=(perf_counter_ns()-start)/1e6
    def identity_now(self):
        r=self.r
        return (id(r.descriptor),id(r.catalog),id(r.caches['contiguous']),r.snapshot,r.namespace,r.weights,
            r.model_weight_identity,r.execution_code_sha256,r.storage,tuple(r.ids),r.coverage,
            r.a.device,r.a.dtype,r.a.capacity_mib)
    def match(self,hits,ids=UNSET,adapter=UNSET,namespace=UNSET,weights=UNSET,device=UNSET,dtype=UNSET):
        if self.identity_now()!=self.identity or hit_sequence(hits)!=self.sequence:
            raise MissingEvidence('oracle exact selection/source/target provenance mismatch')
        expected=(tuple(self.r.ids),'user_a',self.r.namespace,self.r.weights,str(self.r.a.device),'torch.'+self.r.a.dtype)
        actual=(expected[0] if ids is UNSET else tuple(ids) if ids is not None else None,
            adapter if adapter is not UNSET else expected[1],namespace if namespace is not UNSET else expected[2],
            weights if weights is not UNSET else expected[3],str(device) if device is not UNSET else expected[4],str(dtype) if dtype is not UNSET else expected[5])
        if actual!=expected: raise MissingEvidence('oracle target/adapter/model/device/dtype identity mismatch')
    def finish(self,flow):
        self.measure('outside_post_guard_ms',self.r.guard)
        self.measure('outside_post_source_content_ms',lambda:source_content(self.r))
        self.match(flow.hits,flow.ids,flow.user,flow.namespace,flow.weights,flow.device,flow.dtype)
        if self.applicable:
            self.measure('outside_post_snapshot_validation_ms',lambda:self.descriptor.validate(flow.hits))
        if self.mapping_oracle:
            if flow.plan is not self.plan: raise ParityFailure('executed mapping differs from prebound plan')
            self.measure('outside_post_mapping_geometry_ms',lambda:validate_plan(self.r,self.plan,flow.hits))
            if self.measure('outside_post_mapping_seal_ms',lambda:plan_state(self.plan))!=self.plan_seal:
                raise ParityFailure('prepared mapping/indices/backing mutated during oracle request')


PARENTS={
    'request_mapping_select':'hit_selection',
    'prepared_mapping_get':'request_mapping_select',
    'prepared_mapping_oracle_dispatch':'request_mapping_select',
    'source_snapshot_validate':'executor_setup',
    'source_snapshot_oracle_dispatch':'executor_setup',
    'executor_setup':'prefill', 'executor_cleanup':'prefill',
    'fast_index_guard':'selection or outside-request postcheck (scope recorded per observation)',
}


class PathProbe:
    def __init__(self,runtime,arm,lease=None,detailed=False):
        self.r=runtime;self.arm=arm;self.lease=lease;self.detailed=detailed
        self.active=False;self.timer_count=0;self.request_starts=0;self.request_ends=0
        self.calls=Counter();self.host=defaultdict(float);self.flow=None
    @contextmanager
    def span(self,name):
        scope='request_internal' if self.active else 'request_external'
        key=(scope,name);self.calls[key]+=1
        start=perf_counter_ns() if self.detailed else None
        try: yield
        finally:
            if self.detailed:self.host[key]+=(perf_counter_ns()-start)/1e6
    def count(self,name,scope='request_internal'):return self.calls[(scope,name)]
    def require_internal(self):
        if not self.active: raise MissingEvidence('cache-path intervention outside exact request boundary')
    @contextmanager
    def scoped(self):
        import semcache.semcache_engine as em
        from semcache.edgelora.batched_mapping_optimized import SourceBackingDescriptors,PreparedMappingCache
        flow_cls=self.r.b.RequestMapping
        original_timer=em.CPUWallTimer;original_select=flow_cls.select;original_get=PreparedMappingCache.get
        original_validate=SourceBackingDescriptors.validate;original_report=flow_cls.report
        original_context=flow_cls.context
        index_cls=type(self.r.index);original_guard=index_cls.guard
        probe=self
        class RequestTimer:
            def __init__(self):self.timer=original_timer()
            def __getattr__(self,name):return getattr(self.timer,name)
            def __enter__(self):
                self.timer.__enter__();probe.active=True;probe.request_starts+=1;return self
            def __exit__(self,*args):
                self.timer.__exit__(*args);probe.active=False;probe.request_ends+=1
        def timer():
            probe.timer_count+=1
            return RequestTimer() if probe.timer_count==1 else original_timer()
        def select(flow,hits,ids,user):
            with probe.span('request_mapping_select'):
                probe.require_internal()
                if probe.lease:probe.lease.match(hits,ids,user,flow.namespace,flow.weights,flow.device,flow.dtype)
                return original_select(flow,hits,ids,user)
        def get(cache,hits,ids,adapter,namespace,weights,device,dtype,book=None,mode='gather',validate=True):
            if cache is not probe.r.caches['contiguous']:raise MissingEvidence('unexpected mapping cache instance')
            probe.require_internal()
            if probe.lease and probe.lease.mapping_oracle:
                with probe.span('prepared_mapping_oracle_dispatch'):
                    probe.lease.match(hits,ids,adapter,namespace,weights,device,dtype)
                    if mode!='contiguous' or validate is not False:raise MissingEvidence('mapping oracle call boundary/assembly changed')
                    # API compatibility boolean only: record explicitly labels
                    # this as prebound dispatch, not a measured cache lookup hit.
                    return probe.lease.plan,True
            with probe.span('prepared_mapping_get'):
                return original_get(cache,hits,ids,adapter,namespace,weights,device,dtype,book,mode,validate)
        def validate(descriptor,hits,book=None):
            if descriptor is not probe.r.descriptor:raise MissingEvidence('unexpected Source descriptor')
            if probe.active and probe.lease and probe.lease.snapshot_oracle:
                with probe.span('source_snapshot_oracle_dispatch'):
                    probe.lease.match(hits)
                    return 0.  # Legacy elapsed-time return only; no fake measured metric.
            with probe.span('source_snapshot_validate'):return original_validate(descriptor,hits,book)
        def guard(index,*args,**kwargs):
            with probe.span('fast_index_guard'):return original_guard(index,*args,**kwargs)
        @contextmanager
        def context(flow,*args,**kwargs):
            ctx=original_context(flow,*args,**kwargs)
            with probe.span('executor_setup'):audit=ctx.__enter__()
            try:yield audit
            except BaseException:
                import sys
                with probe.span('executor_cleanup'):ctx.__exit__(*sys.exc_info())
                raise
            else:
                with probe.span('executor_cleanup'):ctx.__exit__(None,None,None)
        def report(flow):
            if probe.active:raise MissingEvidence('passive output observation moved inside request timer')
            probe.flow=flow
            return original_report(flow)
        with ExitStack() as stack:
            for obj,name,fn in ((em,'CPUWallTimer',timer),(flow_cls,'select',select),
                (PreparedMappingCache,'get',get),(SourceBackingDescriptors,'validate',validate),
                (flow_cls,'context',context),(flow_cls,'report',report),(index_cls,'guard',guard)):
                stack.enter_context(patch.object(obj,name,fn))
            yield self
            self.verify()
    def verify(self):
        if self.arm=='FULL_NATIVE':
            if self.request_starts or self.flow:raise MissingEvidence('native Full unexpectedly entered cache path')
            return
        if self.active or self.request_starts!=1 or self.request_ends!=1 or self.flow is None or not self.flow.completed:
            raise MissingEvidence('request timing boundary or executor evidence incomplete')
        if self.count('request_mapping_select')!=1 or self.count('executor_setup')!=1 or self.count('executor_cleanup')!=1:
            raise MissingEvidence('unexpected selection/executor invocation count')
        expected=int(bool(self.flow.hits))
        snapshot_skip=int(bool(self.lease and self.lease.snapshot_oracle))
        mapping_skip=int(bool(self.lease and self.lease.mapping_oracle))
        if (self.count('source_snapshot_validate')!=expected-snapshot_skip or
            self.count('source_snapshot_oracle_dispatch')!=snapshot_skip or
            self.count('prepared_mapping_get')!=expected-mapping_skip or
            self.count('prepared_mapping_oracle_dispatch')!=mapping_skip):
            raise MissingEvidence('actual mapping/validation calls differ from approved ablation')
    def audit(self,record):
        positive=bool(record['reused_tokens'])
        plan=self.flow.plan if self.flow is not None else None
        def applicability(which):
            return 'APPLICABLE' if positive else 'NOT_APPLICABLE'
        return dict(timing_boundary_verified=self.arm=='FULL_NATIVE' or self.request_starts==self.request_ends==1,
            snapshot_intervention=applicability('snapshot'),mapping_intervention=applicability('mapping'),
            snapshot_bypassed=bool(self.lease and self.lease.snapshot_oracle),mapping_bypassed=bool(self.lease and self.lease.mapping_oracle),
            calls={scope:{name:n for (s,name),n in self.calls.items() if s==scope} for scope in ('request_internal','request_external')},
            mapping_cache_hit=(None if self.lease and self.lease.mapping_oracle else record.get('mapping_cache_hit')),
            mapping_lookup_kind='PREBOUND_NOT_A_CACHE_LOOKUP' if self.lease and self.lease.mapping_oracle else 'ORIGINAL_CACHE_LOOKUP' if positive else 'NOT_APPLICABLE',
            prebind_cache_hit=self.lease.prebind_cache_hit if self.lease else None,
            prepared_index_usage=(dict(prepared_modes=sorted(plan.prepared_modes),groups=len(plan.groups),
                contiguous_groups=sum(g['is_contiguous'] for g in plan.groups),
                used_index_keys=[['fresh',0]]+[[role,i] for i,g in enumerate(plan.groups) if not g['is_contiguous'] for role in ('source','target')],
                indices=[dict(key=list(k),shape=list(t.shape),dtype=str(t.dtype),device=str(t.device)) for k,t in sorted(plan.index_tensors.items())]) if plan else None),
            excluded_costs_ms=self.lease.costs if self.lease else {},
            snapshot_boundary='executor entry only; retained FastOccurrenceIndex.guard during actual selection and post-request checks',
            qkv_memory_counters=record.get('memory',{}).get('temporary'))
    def output_hashes(self):
        if self.active:raise MissingEvidence('QKV equality audit attempted inside request')
        if self.flow is None:return None
        expected=len(self.r.adapter.layers)*3
        result={f'{layer}:{role}':dict(sha256=tensor_sha(t),shape=list(t.shape),dtype=str(t.dtype),device=str(t.device))
            for layer,values in self.flow.audit.projections.items() for role,t in values.items()}
        if len(result)!=expected:raise MissingEvidence('missing actual assembled QKV outputs')
        return result
    def export(self):
        return [dict(component=name,location=scope,host_ms=self.host.get((scope,name)),cuda_ms=None,calls=n,
            parent=PARENTS.get(name),inclusive=True,clock='cpu_perf_counter_ns',
            status='MEASURED' if self.detailed else 'COUNTS_ONLY',do_not_sum_nested=True)
            for (scope,name),n in sorted(self.calls.items())]
