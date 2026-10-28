"""Reuse R2-A assembly ranges, with policy insertion observed only in profiling."""
from contextlib import contextmanager
import json
from unittest.mock import patch
from semcache.diagnostics.r2a_profiler import require_profiler,ranges,operator_rows,trace_rows,ProfilerUnavailable


@contextmanager
def extra_ranges():
    import torch
    from semcache.cache.global_cache import GlobalCache
    insert=GlobalCache.insert
    def policy(*args,**kwargs):
        with torch.profiler.record_function('r2a::cache_policy_insert_inclusive'): return insert(*args,**kwargs)
    # V2 imports batched_mixed_projection.span at context entry. R2-A's ranges()
    # already wraps that actual symbol, so no duplicate assembly range is added.
    with patch.object(GlobalCache,'insert',policy): yield


def collect(runtime,mode,path,max_trace_mib=64):
    import torch
    from semcache.metrics.r1_runtime import synchronize
    p,activities=require_profiler(runtime.a.device)
    reference,_=runtime.invoke(mode)
    with p.profile(activities=activities,record_shapes=True,profile_memory=True,with_stack=False) as prof:
        with ranges(runtime),extra_ranges(),p.record_function('r2a::request/'+mode):
            logits,record=runtime.invoke(mode)
        synchronize(runtime.a.device)
    if not torch.equal(reference,logits): raise RuntimeError('PROFILE_PARITY_FAILED: logits changed')
    prof.export_chrome_trace(str(path))
    if path.stat().st_size>max_trace_mib*1024**2:
        raise RuntimeError('TRACE_BUDGET_EXCEEDED: trace preserved; reduce profile scope')
    kernels,definition=trace_rows(json.loads(path.read_text()))
    return dict(status='PROFILE_COLLECTED',instrumented_request_ms=record['request_ms'],**definition,
        memory=record.get('memory'),memory_accounting_note=record.get('memory_accounting_note'),coverage=record['coverage_audit'],
        scope='separate instrumented trial; parent/child intervals not summed; invocation range includes outside-request clone/audits',
        limits='GPU activity correlation only; no occupancy/Tensor Core/cuBLAS algorithm inference'),operator_rows(prof.events()),kernels
