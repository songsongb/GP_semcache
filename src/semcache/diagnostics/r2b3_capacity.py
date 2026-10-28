"""Capacity observation and exact per-case approval; no automatic reduced workload."""
import gc
from pathlib import Path
import traceback


def host_memory():
    try:
        fields={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/self/status').read_text().splitlines()
            if s.startswith(('VmRSS:','VmHWM:'))}
        return dict(status='MEASURED',rss_bytes=fields['VmRSS'],peak_rss_bytes=fields['VmHWM'])
    except Exception as exc: return dict(status='UNAVAILABLE',rss_bytes=None,peak_rss_bytes=None,reason=str(exc))


def memory(device,reset=False):
    from semcache.metrics.r1_runtime import memory_snapshot
    if reset and device.startswith('cuda'):
        import torch
        torch.cuda.reset_peak_memory_stats(device)
    return dict(host=host_memory(),gpu=memory_snapshot(device))


def failure_record(exc,context):
    status='MEMORY_PREFLIGHT_FAILED' if isinstance(exc,MemoryError) else 'OOM' if 'out of memory' in str(exc).lower() else 'INVALID_EXECUTION'
    return dict(status=status,context=context,failure_type=type(exc).__name__,failure=str(exc),
        failure_code=getattr(exc,'code',None),traceback=traceback.format_exc(),valid_performance_evidence=False)


def release_failed_case(runtime):
    # Called only after an unsupported capacity case; never in a request timer.
    runtime.snapshot=None; runtime.source_logits=None
    gc.collect()
    if runtime.device.startswith('cuda'):
        import torch
        torch.cuda.synchronize(runtime.device); torch.cuda.empty_cache()


def approved(pilots,case_id,arm,environment_identity):
    for pilot in pilots:
        if pilot.get('experiment')!='R2-B3' or pilot.get('phase')!='pilot': continue
        if pilot.get('execution_status') not in ('COMPLETE','COMPLETE_WITH_FAILURES') or not pilot.get('model_weights_verified'): continue
        if pilot.get('environment_identity')!=environment_identity: continue
        evidence=pilot['conditions'].get(case_id+'_'+arm)
        if evidence and evidence['status']=='CAPACITY_VALIDATED' and evidence.get('structural_valid') and evidence.get('cache_integrity_valid'):
            return evidence
    raise ValueError(f'PILOT_REQUIRED: no matching validated {case_id}/{arm} on this code/model/adapter/GPU/seed/endpoint')
