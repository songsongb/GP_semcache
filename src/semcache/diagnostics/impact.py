"""M9-A.2 same-formula alternatives and disjoint wall/CUDA attribution.

No production reducer is replaced. Arithmetic is shared with opt-in R1-B1;
the diagnostic still owns its original timing and history finalization.
CURRENT_BLOCKWISE mirrors the production reducer's operation order;
the benchmark additionally checks it against the production reducer on the same
attention tensors. FP64 arithmetic is intentional: it is the existing reducer's
explicit promotion, not a new precision optimization.
"""
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter_ns
import math

from semcache.cache.attention_impact import PaperRowL2SumReducer
from semcache.cache.metric_manager import QueryImpactHistory

from semcache.cache.impact_execution import MODES, PARTS, compute_window_values, _validate, _mask
ATOL, RTOL = 1e-10, 1e-12


class Attribution:
    def __init__(self, device, enabled):
        self.enabled = enabled
        self.cuda = str(device).startswith('cuda') and enabled
        self.parts = dict.fromkeys(PARTS, 0.)
        self.counts = defaultdict(int)
        self.events = []

    @contextmanager
    def stage(self, field, device_work=False):
        if not self.enabled:
            yield
            return
        start = perf_counter_ns()
        pair = None
        if device_work and self.cuda:
            import torch
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            pair[0].record()
        yield
        if pair:
            pair[1].record()
            self.events.append((field, pair))
        self.parts[field] += (perf_counter_ns()-start)/1e6

    def resolved(self):
        gpu = defaultdict(float)
        if self.events:
            # All host results are already available. This final event boundary is
            # outside impact_total_ms, solely for attribution event bookkeeping.
            self.events[-1][1][1].synchronize()
            self.counts['attribution_event_synchronize_outside_total'] += 1
            for field, (start, end) in self.events:
                gpu[field.replace('_ms', '_device_ms')] += start.elapsed_time(end)
        return dict(gpu)


def parity(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError('Impact vector lengths differ')
    errors = [abs(a-b) for a, b in zip(reference, candidate)]
    denom = math.sqrt(sum(x*x for x in reference))
    norm = math.sqrt(sum(e*e for e in errors))
    return dict(impact_value_max_absolute_error=max(errors, default=0.),
        impact_value_relative_l2_error=norm/denom if denom else (0. if not norm else None),
        impact_values_within_tolerance=all(math.isfinite(b) and abs(a-b) <= ATOL+RTOL*abs(a)
                                         for a, b in zip(reference, candidate)), atol=ATOL, rtol=RTOL)


def finalize(windows, values, cluster, history):
    """Same repeated-key mean and one query observation as SemCacheEngine."""
    grouped, spans = defaultdict(list), {}
    for window, value in zip(windows, values):
        spans[window.start] = value
        grouped[(cluster, window.token_ids)].append(value)
    per_query = {key: sum(items)/len(items) for key, items in grouped.items()}
    history.append(cluster, 'impact_audit', 2, per_query)
    return spans, per_query


def measure_impact(attentions, windows, mode, valid_attention_mask=None, *,
                   cluster=0, history=None, instrument=True, production_reference=False):
    """Measure reducer + engine-equivalent grouping/history, excluding policy.

    Wall stages account for CPU launch time. Scalar/.cpu waits include outstanding
    GPU work: device_to_host_ms is NOT a pure PCIe transfer measurement. CUDA event
    durations are additional overlapping diagnostics and must not be added to wall.
    Callers synchronize the completed forward before entering this function.
    """
    import torch
    if mode not in MODES:
        raise ValueError('Unknown impact implementation')
    if production_reference and mode != MODES[0]:
        raise ValueError('Production reference is CURRENT_BLOCKWISE only')
    windows = list(windows)
    n = attentions[0].shape[-1] if attentions else 0
    if windows and (not attentions or any(not 0 <= w.start < w.end <= n for w in windows)):
        raise ValueError('Invalid attention/window dimensions')
    device = attentions[0].device if attentions else 'cpu'
    timer = Attribution(device, instrument)
    history = history if history is not None else QueryImpactHistory()
    metadata = dict(candidate_block_count=len(windows), token_count=n, layer_count=len(attentions),
        head_count=attentions[0].shape[1] if attentions else 0,
        attention_tensor_shapes=[list(a.shape) for a in attentions],
        attention_tensor_shape=list(attentions[0].shape) if attentions else None,
        device=str(device), dtype=str(attentions[0].dtype) if attentions else None,
        reduction_dtype='torch.float64', input_attention_modified=False,
        implementation=mode, provenance='MEASURED' if mode == MODES[0] else 'MEASURED_RESEARCH_EXTENSION',
        formula_provenance='PAPER_DEFINED', interpretation_provenance='REPRODUCTION_CHOICE',
        formula=PaperRowL2SumReducer().metadata['formula'])
    start_time = perf_counter_ns()
    values_out = compute_window_values(attentions, windows, mode, valid_attention_mask,
        timer=timer, production_reference=production_reference)
    with timer.stage(PARTS[4]):
        spans, per_query = finalize(windows, values_out, cluster, history)
    total_ms = (perf_counter_ns()-start_time)/1e6
    # Dispatch/loop/context-manager overhead belongs to finalize/residual, not to
    # device arithmetic. Expose it separately rather than pretending it is free.
    residual = total_ms-sum(timer.parts.values())
    timer.parts[PARTS[4]] += residual
    event_ms = timer.resolved() if instrument else {}
    timer.counts['cuda_event_records'] = 2*len(timer.events)
    counts = {k: v for k, v in timer.counts.items() if v}
    sync_reads = counts.get('finite_scalar_host_reads', 0)+counts.get('impact_scalar_host_reads', 0)
    return dict(**metadata, **timer.parts, **event_ms, impact_total_ms=total_ms,
        impact_unscoped_host_overhead_ms=residual, instrumentation_enabled=instrument,
        operation_counts=counts, scalar_or_result_cuda_sync_points=(sync_reads+counts.get('batched_host_materializations', 0)
                                                          if str(device).startswith('cuda') else 0),
        synchronization_scope='bool(finite)/item per block-layer or one packed cpu transfer; '
            'blocking CPU-mask to CUDA copies and window-index construction also wait; '
            'attribution final event sync outside total; forward boundary sync outside total',
        wall_timing_scope='launch/host/wait partitions; D2H includes queued-device completion',
        cuda_timing_scope='overlapping attribution only; never sum with wall components',
        impact_values=values_out, per_query_impact_count=len(per_query))
