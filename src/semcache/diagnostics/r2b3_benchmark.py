"""B3 adapter around frozen B2 request timing; numeric checks remain outside timers."""
from contextlib import contextmanager
from unittest.mock import patch
from semcache.diagnostics import r2b2_benchmark as b2
from semcache.diagnostics.r1b3_fp16_divergence import logit_metrics
from semcache.diagnostics.r2b1_prefix_cache import PrefixError
from semcache.diagnostics.r2b3_workloads import validate_case

Runtime=b2.B2Runtime


def strict_scalar(reference,actual):
    """Dependency-free specification, reference-relative tolerance (unchanged 1e-3)."""
    import math
    if not reference or len(reference)!=len(actual): raise ValueError('equal nonempty arrays required')
    if not all(math.isfinite(x) for x in [*reference,*actual]): raise ValueError('nonfinite')
    failed=[i for i,(ref,a) in enumerate(zip(reference,actual)) if abs(a-ref)>1e-3+1e-3*abs(ref)]
    return dict(allclose=not failed,tolerance_violations=len(failed),elements=len(reference))


def strict_metrics(reference,actual):
    import torch
    if reference.shape!=actual.shape or reference.dtype!=actual.dtype:
        raise PrefixError('REJECT_LOGIT_SHAPE','identical shape/dtype required; no broadcasting')
    # Preserve legacy descriptive metrics, explicitly replace strict decision with
    # the specified reference-relative elementwise predicate. Double precision
    # here evaluates differences of already-computed logits, not model inference.
    result=logit_metrics(reference,actual)
    result['legacy_b2_allclose']=result['allclose']
    ref=reference.detach().double(); a=actual.detach().double()
    failed=(~torch.isfinite(ref)) | (~torch.isfinite(a)) | ((a-ref).abs()>1e-3+1e-3*ref.abs())
    count=int(failed.sum()); result.update(allclose=count==0,tolerance_violations=count,
        tolerance_violation_fraction=count/ref.numel(),elements=ref.numel(),
        tolerance_definition='abs(actual-reference)<=1e-3+1e-3*abs(reference)',
        correctness_status='STRICT_PASS' if count==0 else 'STRICT_FAIL')
    result['legacy_b2_first_tolerance_failure']=result.get('first_tolerance_failure')
    locations=failed.nonzero()
    result['first_tolerance_failure']=dict(index=locations[0].tolist()) if count else None
    return result


def quality(runtime,full,warm):
    if runtime.endpoint['effective']!='b1_all_logits': raise PrefixError('REJECT_ENDPOINT','B3 primary requires b1_all_logits')
    suffix=full[:,runtime.p:,:]
    q=strict_metrics(suffix,warm); last=strict_metrics(full[:,-1:,:],warm[:,-1:,:])
    return dict(suffix=q,last_position=last,strict=q['allclose'] and last['allclose'],
        scope='Full[:,P:L,:] versus entire Warm suffix; last token separately')


@contextmanager
def numeric_contract():
    # Scoped correctness callback only. No model/timer/cache/forward patch.
    with patch.object(b2,'quality',quality): yield


def gate(runtime):
    validate_case(runtime.n,runtime.p,runtime.model.config.max_position_embeddings)
    with numeric_contract(): return b2.structural_gate(runtime)


def run_pairs(runtime,args,identity,save):
    with numeric_contract(): return b2.run_pairs(runtime,args,identity,save)


def enrich(rows,runtime,phase,source):
    geometry=validate_case(runtime.n,runtime.p,runtime.model.config.max_position_embeddings)
    return [dict(r,**geometry,benchmark_phase=phase,output_matched=False,materialization_confounded=True,
        endpoint_classification='B1_COMPATIBLE_LATENCY_OBSERVATION',
        cache_bytes=source.get('cache_bytes'),source_prefill_ms=source.get('prefix_prefill_ms'),snapshot_ms=source.get('snapshot_ms'),
        prepare_wall_ms=None,prepare_wall_status='SEPARATE_COMPONENT_TRIAL_IN_SOURCE_COSTS',
        normal_wall_ms=r['normal_warm_wall_ms'],full_normal_speedup=r['full_wall_ms']/r['normal_warm_wall_ms'],
        strict_correctness=r['correctness_status'],structural_reuse_valid=True,cache_integrity_valid=True,
        gpu_memory_peak=r['memory']) for r in rows]
