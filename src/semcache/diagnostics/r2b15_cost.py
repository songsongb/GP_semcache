"""B1.5 cost isolation. Normal uses frozen B1; prepared is single-use IDEAL_DIAGNOSTIC."""
from contextlib import contextmanager,ExitStack
from time import perf_counter_ns
from unittest.mock import patch
import uuid
from semcache.diagnostics import r2b1_prefix_cache as cache_ops
from semcache.diagnostics.r2b1_prefix_cache import PrefixError
from semcache.metrics.r1 import digest


def cache_stamps(cache):
    return tuple((id(t),cache_ops.tensor_stamp(t)) for pair in cache_ops.cache_pairs(cache)[0] for t in pair)


def cache_hashes(cache):
    return tuple(cache_ops.tensor_hash(t) for pair in cache_ops.cache_pairs(cache)[0] for t in pair)


class PreparedTicket:
    def __init__(self,runtime):
        from semcache.metrics.r1_runtime import synchronize
        if runtime.snapshot is None or runtime.p<=0: raise PrefixError('REJECT_PREPARED_CACHE','nonempty independent source snapshot required')
        self.r=runtime; self.state='BUILDING'; self.calls=0; self.id=uuid.uuid4().hex
        runtime.check(); self.binding=self.current_binding()
        synchronize(runtime.device); start=perf_counter_ns()
        self.cache=runtime.snapshot.materialize(runtime.identity(),runtime.ids,position_start=runtime.p,mask=[1]*runtime.n)
        synchronize(runtime.device); self.excluded_prepare_wall_ms=(perf_counter_ns()-start)/1e6
        runtime.snapshot.inspect_request(self.cache)
        self.stamps=cache_stamps(self.cache); self.hashes=cache_hashes(self.cache); self.state='READY'
    def current_binding(self):
        r=self.r
        return (id(r.snapshot),r.snapshot.identity_hash,tuple(r.ids),r.p,r.n,r.identity())
    def check(self):
        if self.state!='READY': raise PrefixError('REJECT_PREPARED_REUSE','ticket no longer ready: '+self.state)
        try:
            self.r.check()
            if self.current_binding()!=self.binding or cache_stamps(self.cache)!=self.stamps or cache_hashes(self.cache)!=self.hashes:
                raise PrefixError('REJECT_PREPARED_MUTATION','target/snapshot/namespace/request cache changed')
            self.r.snapshot.inspect_request(self.cache)
        except Exception:
            self.state='INVALIDATED'; raise
    def consume(self,identity,ids,*,position_start,mask):
        if self.state!='READY' or self.calls: raise PrefixError('REJECT_PREPARED_REUSE','prepared cache is single-use')
        cache_ops.validate_request(self.r.snapshot.identity,identity,self.r.snapshot.prefix_ids,ids,position_start=position_start,mask=mask)
        if self.current_binding()!=self.binding or cache_stamps(self.cache)!=self.stamps:
            raise PrefixError('REJECT_PREPARED_MUTATION','prepared request changed after prior verification')
        cache_ops.validate_pairs(cache_ops.cache_pairs(self.cache)[0],**self.r.snapshot.layout)
        self.state='CONSUMED'; self.calls+=1
        return self.cache
    @contextmanager
    def use(self):
        self.check()
        try:
            # One instance method only; original B1 normal path is unchanged.
            with patch.object(self.r.snapshot,'materialize',self.consume): yield self
            if self.calls!=1: raise PrefixError('MISSING_PREPARED_EVIDENCE','expected one prepared materialize handoff')
        except BaseException:
            self.state='DISCARDED'; raise
    def report(self):
        if self.state!='CONSUMED' or self.calls!=1: raise PrefixError('MISSING_PREPARED_EVIDENCE','ticket did not complete one handoff')
        after=[int(pair[0].shape[2]) for pair in cache_ops.cache_pairs(self.cache)[0]]
        return dict(id=self.id,state=self.state,materialize_handoffs=self.calls,excluded_prepare_wall_ms=self.excluded_prepare_wall_ms,
            input_length_before=self.r.p,input_lengths_after=after,source_length=self.r.snapshot.length,
            snapshot_identity=self.r.snapshot.identity_hash,clone_bytes_outside_request=self.r.snapshot.bytes,
            classification='IDEAL_DIAGNOSTIC',exclusion='only clone_tensor loop + cache factory creation moved before request; same namespace and shape guard retained',
            guard='content check outside timer; pointer/version/namespace single-use check inside; serial non-adversarial fixture')


def invoke(runtime,mode,*,audit=False,book=None):
    if mode in ('FULL_USE_CACHE','WARM_NORMAL'):
        with instrument(runtime,book) if book else _nothing():
            return runtime.forward('FULL_USE_CACHE' if mode=='FULL_USE_CACHE' else 'EXACT_PREFIX_WARM',audit=audit),None
    if mode!='WARM_PREPARED_DIAGNOSTIC': raise ValueError(mode)
    ticket=PreparedTicket(runtime)
    with ticket.use():
        # Optional detailed ranges are installed AFTER independent preparation,
        # so excluded preparation cannot leak into the prepared request's spans.
        with instrument(runtime,book) if book else _nothing(): result=runtime.forward('EXACT_PREFIX_WARM',audit=audit)
    runtime.check()
    return result,ticket.report()


@contextmanager
def _nothing(): yield


class CostBook:
    """Instrumented clocks only. CUDA intervals resolve after enclosing request sync."""
    def __init__(self,device): self.device=device; self.host={}; self.events={}; self.calls={}; self.clone_bytes=0
    @contextmanager
    def span(self,name,gpu=False):
        import torch
        events=None
        if gpu and str(self.device).startswith('cuda'):
            with torch.cuda.device(self.device):
                events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)); events[0].record()
        start=perf_counter_ns()
        try: yield
        finally:
            self.host[name]=self.host.get(name,0.)+(perf_counter_ns()-start)/1e6
            self.calls[name]=self.calls.get(name,0)+1
            if events:
                with torch.cuda.device(self.device): events[1].record()
                self.events.setdefault(name,[]).append(events)
    def export(self):
        relations={
            'adapter_selection':('request_wall','child CPU scope'),
            'eval_mode':('request_wall','child CPU scope'),
            'input_mask_position':('request_wall','child; GPU intervals include launch gaps'),
            'cache_materialize':('request_wall','inclusive; contains namespace/clone/container/shape guards'),
            'namespace_guard':('cache_materialize','nested CPU; do not add to inclusive parent'),
            'kv_clone':('cache_materialize','sum of actual tensor clones; no parent+child addition'),
            'cache_container':('cache_materialize','nested factory; may allocate/copy again'),
            'native_forward':('request_wall','includes attention/FFN/LM head/output cache'),
            'lm_head':('native_forward','nested, do not add to native_forward'),
        }
        rows=[]
        for name,(parent,relation) in relations.items():
            rows.append(dict(component=name,parent=parent,nesting=relation,host_ms=self.host.get(name),calls=self.calls.get(name),
                cuda_interval_ms=sum(a.elapsed_time(b) for a,b in self.events[name]) if name in self.events else None,
                status='MEASURED' if name in self.host else 'NOT_EXECUTED',instrumentation='breakdown',
                clock_definition='host submission wall / CUDA stream interval; neither is profiler self time or kernel busy time',
                clone_bytes=self.clone_bytes if name=='kv_clone' and name in self.host else None,
                kernel_launches=None,memcpy_events=None,allocator_bytes=None))
        rows.append(dict(component='return_cleanup',parent='invocation_outside_request',host_ms=None,cuda_interval_ms=None,
            status='UNSUPPORTED_ISOLATED_BOUNDARY',instrumentation='breakdown',
            nesting='B1 closes request before cache audits/logit CPU export and frame cleanup; no fake cleanup measurement'))
        return rows


@contextmanager
def instrument(runtime,book):
    """Read-only scoped function wrappers; no extra synchronization or arithmetic changes."""
    with ExitStack() as stack:
        def wrap(owner,name,label,gpu=False):
            fn=getattr(owner,name)
            def called(*args,**kwargs):
                with book.span(label,gpu): return fn(*args,**kwargs)
            stack.enter_context(patch.object(owner,name,called))
        wrap(runtime.model,'set_adapter','adapter_selection'); wrap(runtime.model,'eval','eval_mode')
        wrap(runtime.api,'kwargs','input_mask_position',True)
        wrap(runtime.model,'forward','native_forward',True)
        if getattr(runtime.base,'lm_head',None) is not None: wrap(runtime.base.lm_head,'forward','lm_head',True)
        wrap(cache_ops,'validate_request','namespace_guard')
        if runtime.snapshot:
            wrap(runtime.snapshot,'materialize','cache_materialize',True)
            if runtime.snapshot.factory: wrap(runtime.snapshot,'factory','cache_container',True)
        clone=cache_ops.clone_tensor
        def cloned(t):
            book.clone_bytes+=t.numel()*t.element_size()
            with book.span('kv_clone',True): return clone(t)
        stack.enter_context(patch.object(cache_ops,'clone_tensor',cloned))
        yield book


def prepare_only(runtime,book=None):
    from semcache.metrics.r1_runtime import synchronize
    runtime.check()
    with instrument(runtime,book) if book else _nothing():
        synchronize(runtime.device); started=perf_counter_ns()
        cache=runtime.snapshot.materialize(runtime.identity(),runtime.ids,position_start=runtime.p,mask=[1]*runtime.n)
        synchronize(runtime.device); wall=(perf_counter_ns()-started)/1e6
    runtime.snapshot.inspect_request(cache); runtime.check()
    return dict(mode='PREPARE_ONLY',wall_ms=wall,cache_bytes=runtime.snapshot.bytes,
        native_forward_calls=0,scope='isolated snapshot materialize; namespace+clones+factory+shape checks; no model forward',
        input_cache_api=cache_ops.cache_pairs(cache)[1],instrumentation='breakdown' if book else 'component_minimal')


def structural_gate(runtime):
    import torch
    from semcache.diagnostics.r2b1_prefix_audit import compare_suffix
    runtime.check(); full=runtime.forward('FULL_USE_CACHE',audit=True)
    normal=runtime.forward('EXACT_PREFIX_WARM',audit=True)
    prepared,ticket=invoke(runtime,'WARM_PREPARED_DIAGNOSTIC',audit=True)
    if not torch.equal(normal['logits'],prepared['logits']) or normal['audit']['rows']!=prepared['audit']['rows']:
        raise PrefixError('REJECT_PREPARED_PARITY','normal/prepared outputs or physical row audit differ')
    return dict(status='PHYSICAL_PREFIX_SKIP_VERIFIED',quality=compare_suffix(full['logits'],normal['logits'],runtime.p),
        full=full['audit'],normal=normal['audit'],prepared=prepared['audit'],prepared_parity_exact=True,ticket=ticket,canonical=runtime.check())


def monitoring(runtime):
    """Block-boundary monitoring; no CUDA synchronization or guessed GPU index."""
    import subprocess,os
    from datetime import datetime,timezone
    state=runtime.gpu_state()
    state['timestamp_utc']=datetime.now(timezone.utc).isoformat()
    state['monitoring_status']='MONITORING_UNAVAILABLE' if state.get('reason') or not state.get('gpu_uuid') else 'MEASURED'
    state['other_processes']=None
    if state.get('gpu_uuid'):
        try:
            p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
            state['other_processes']=[line for line in p.stdout.splitlines() if len(line.split(','))>=2
                and line.split(',')[0].strip()==state['gpu_uuid'] and line.split(',')[1].strip()!=str(os.getpid())]
            state['process_scope']='same GPU UUID, excluding this process'
        except Exception as exc:
            state['monitoring_status']='MONITORING_UNAVAILABLE'; state['process_monitoring_reason']=str(exc)
    return state
