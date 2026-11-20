"""L2C candidate: defer host reads only; frozen Strict reducer is untouched.

No token precompute, prefix sum, window removal or cross-layer reassociation.
Native CPU parity is mandatory in candidate CPU approvals; GPU parity remains
separate. The default Light implementation stays the frozen reference.
"""
import math
import time

from .communication import require
from .r5_subsequence_generation import StreamingImpact
from .r5_subsequence_profile import call

REFERENCE = 'REFERENCE_WINDOW_SCALAR'
HOST_BATCH = 'DEFERRED_HOST_PER_LAYER_V1'
IMPLEMENTATIONS = (REFERENCE, HOST_BATCH)
MAX_WINDOWS = 512
STAGING_RESERVE_BYTES = 4 * 2**20  # Conservative extra scalar/packing allocator reserve, not measured VRAM.


def implementation(plan):
    value = plan.get('impact_implementation', REFERENCE)
    require(value in IMPLEMENTATIONS, 'unknown impact implementation')
    return value


def resource_bounds(resources, mode):
    require(mode in IMPLEMENTATIONS, 'unknown impact implementation')
    if mode == REFERENCE:
        return resources
    result = dict(resources, reasons=list(resources['reasons']))
    result['impact_batch_staging_reserve_bytes'] = STAGING_RESERVE_BYTES
    result['impact_layer_working_bytes'] += STAGING_RESERVE_BYTES
    result['gpu_estimate_bytes'] += STAGING_RESERVE_BYTES
    if (result['impact_layer_working_bytes'] > 2**30 or
            result['gpu_estimate_bytes'] >= result['gpu_allocation_ceiling_bytes']):
        result['reasons'].append('IMPACT_BATCH_STAGING_MEMORY_BOUND')
        result['status'] = 'BLOCKED_INFEASIBLE'
    return result


def deferred_window_values(attention, windows, valid_attention_mask=None):
    """Same one-layer/window operations as PaperRowL2SumReducer.reduce().

    Finite predicates and FP64 scalars remain on-device until one packed host
    materialization. Invalid valid cells fail before any result/history/admission
    is returned, though later window kernels may already have been queued.
    """
    windows = list(windows)
    require(len(windows) <= MAX_WINDOWS, 'impact scalar staging window bound exceeded')
    if not windows:
        return []
    import torch
    if (attention is None or attention.ndim != 4 or attention.shape[0] != 1
            or attention.shape[1] < 1 or attention.shape[2] != attention.shape[3]):
        raise ValueError('Expected batch-one square prefill attention covering query span')
    finite_flags, scalars = [], []
    for window in windows:
        start, end = window.start, window.end
        if not 0 <= start < end <= attention.shape[2]:
            raise ValueError('Need attention tensors and a nonempty query span')
        # Keep these arithmetic shapes/order identical to the frozen reducer.
        values = attention.detach().double()[0, :, start:end, :]
        n = attention.shape[-1]
        valid = (torch.arange(n, device=values.device)[None, :]
                 <= torch.arange(start, end, device=values.device)[:, None])
        if valid_attention_mask is not None:
            mask = valid_attention_mask.to(device=values.device, dtype=torch.bool)
            if mask.ndim == 2 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim != 1 or len(mask) != n:
                raise ValueError('Valid mask must cover prefill positions')
            valid &= mask[start:end, None] & mask[None, :]
        values = values.masked_fill(~valid.unsqueeze(0), 0)
        finite_flags.append(torch.isfinite(values).all())
        scalars.append(values.norm(dim=-1).mean(dim=0).sum())
    # Stack/convert booleans only; do not reduce/reassociate Impact scalars.
    packed = torch.stack((torch.stack(finite_flags).to(dtype=torch.float64), torch.stack(scalars)))
    flags, host_values = packed.cpu().tolist()
    output = []
    for ok, value in zip(flags, host_values):
        if not ok:
            raise ValueError('Nonfinite valid attention')
        value = 0.0 + value  # Same Python initialization/addition as one-layer reference.
        if not math.isfinite(value):
            raise ValueError('Nonfinite attention impact')
        output.append(value)
    return output


class DeferredHostImpact(StreamingImpact):
    """Original per-layer streaming lifecycle with one host batch per layer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host_materialization_calls = 0

    def observe(self, layer, prob):
        require(layer == len(self.observed) and layer < self.layers, 'attention layer order differs')
        n = len(self.req['token_ids'])
        require(prob is not None and len(prob.shape) == 4 and prob.shape[0] == 1 and
                tuple(prob.shape[2:]) == (n, n), 'actual square prefill probabilities required')
        require(len(self.windows) <= MAX_WINDOWS and all(0 < w.end - w.start <= 3 for w in self.windows),
                'bounded original impact windows required')
        size = prob.numel() * prob.element_size()
        require(size <= self.max_layer_bytes, 'single-layer attention budget exceeded')
        working_bytes = size + prob.numel() * 8 + prob.shape[1] * 3 * n * 8 + STAGING_RESERVE_BYTES
        require(working_bytes <= 2**30, 'single-layer impact working set exceeds1GiB estimate')
        start = time.monotonic_ns()
        double = call('impact_fp64_conversion', lambda: prob.detach().double(), cuda=True)
        reduced = call('actual_attention_impact_reduction', deferred_window_values, double, self.windows, cuda=True)
        values = {w.start: v for w, v in zip(self.windows, reduced)}
        self.host_materialization_calls += int(bool(self.windows))
        # Keep the frozen per-layer Python accumulation and record format.
        for key, value in values.items():
            self.values[key] += value
        item = None if self.evidence is None else call('impact_evidence_save_inclusive', self.evidence.save, prob,
                                                       self.req['request_id'] + f'_impact_l{layer}')
        self.elapsed_ms += (time.monotonic_ns() - start) / 1e6
        self.peak_bytes = max(self.peak_bytes, size)
        self.peak_working_estimate = max(self.peak_working_estimate, working_bytes)
        self.records.append(dict(layer=layer, probabilities=item, span_values=values))
        self.observed.append(layer)


def create_impact(req, layers, windows, mode=REFERENCE):
    require(mode in IMPLEMENTATIONS, 'unknown impact implementation')
    cls = StreamingImpact if mode == REFERENCE else DeferredHostImpact
    return cls(req, layers, windows, evidence=None)
