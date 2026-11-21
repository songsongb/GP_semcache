"""L2E: reuse layer-local mask operands; keep every window reduction unchanged.

The frozen reducer and L2C implementation remain separate. No row precomputation,
prefix sum, window reordering, full NxN mask, or extra host scalar read.
"""
import math
import time

from .communication import require
from .r5_d5_impact import DeferredHostImpact, MAX_WINDOWS, STAGING_RESERVE_BYTES, MASK_INDEX_REUSE
from .r5_subsequence_profile import call

def index_reserve_bytes(n):
    """Int64 positions plus a possible bool padding copy; round up to 4 KiB.

    The normal Light plan bounds N<=512. Runtime still accounts for actual N,
    including direct CPU tests outside that plan, without weakening old guards.
    """
    return ((9 * n + 4095) // 4096) * 4096


INDEX_RESERVE_BYTES = index_reserve_bytes(512)


def reused_window_values(attention, windows, valid_attention_mask=None):
    """Same FP64 operands and norm/mean/sum as deferred_window_values().

    Allocate positions and convert/validate padding once, lazily after the first
    window's shape/bounds checks. All shared operands are read-only. No cached
    state survives this invocation/layer. Packed flags/scalars are read once.
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
    positions, mask = None, None
    for window in windows:
        start, end = window.start, window.end
        if not 0 <= start < end <= attention.shape[2]:
            raise ValueError('Need attention tensors and a nonempty query span')
        values = attention.detach().double()[0, :, start:end, :]
        n = attention.shape[-1]
        if positions is None:
            positions = torch.arange(n, device=values.device)
            if valid_attention_mask is not None:
                mask = valid_attention_mask.to(device=values.device, dtype=torch.bool)
                if mask.ndim == 2 and mask.shape[0] == 1:
                    mask = mask[0]
                if mask.ndim != 1 or len(mask) != n:
                    raise ValueError('Valid mask must cover prefill positions')
        valid = positions[None, :] <= positions[start:end, None]
        if mask is not None:
            valid &= mask[start:end, None] & mask[None, :]
        values = values.masked_fill(~valid.unsqueeze(0), 0)
        finite_flags.append(torch.isfinite(values).all())
        scalars.append(values.norm(dim=-1).mean(dim=0).sum())
    packed = torch.stack((torch.stack(finite_flags).to(dtype=torch.float64), torch.stack(scalars)))
    flags, host_values = packed.cpu().tolist()
    output = []
    for ok, value in zip(flags, host_values):
        if not ok:
            raise ValueError('Nonfinite valid attention')
        value = 0.0 + value
        if not math.isfinite(value):
            raise ValueError('Nonfinite attention impact')
        output.append(value)
    return output


class IndexReuseImpact(DeferredHostImpact):
    """Original streaming lifecycle, L2C host batching, layer-local operand reuse."""
    def observe(self, layer, prob):
        require(layer == len(self.observed) and layer < self.layers, 'attention layer order differs')
        n = len(self.req['token_ids'])
        require(prob is not None and len(prob.shape) == 4 and prob.shape[0] == 1 and
                tuple(prob.shape[2:]) == (n, n), 'actual square prefill probabilities required')
        require(len(self.windows) <= MAX_WINDOWS and all(0 < w.end - w.start <= 3 for w in self.windows),
                'bounded original impact windows required')
        size = prob.numel() * prob.element_size()
        require(size <= self.max_layer_bytes, 'single-layer attention budget exceeded')
        working_bytes = (size + prob.numel() * 8 + prob.shape[1] * 3 * n * 8 +
                         STAGING_RESERVE_BYTES + index_reserve_bytes(n))
        require(working_bytes <= 2**30, 'single-layer impact working set exceeds1GiB estimate')
        start = time.monotonic_ns()
        double = call('impact_fp64_conversion', lambda: prob.detach().double(), cuda=True)
        reduced = call('actual_attention_impact_reduction', reused_window_values, double, self.windows, cuda=True)
        values = {w.start: v for w, v in zip(self.windows, reduced)}
        self.host_materialization_calls += int(bool(self.windows))
        for key, value in values.items():
            self.values[key] += value
        item = None if self.evidence is None else call('impact_evidence_save_inclusive', self.evidence.save, prob,
                                                       self.req['request_id'] + f'_impact_l{layer}')
        self.elapsed_ms += (time.monotonic_ns() - start) / 1e6
        self.peak_bytes = max(self.peak_bytes, size)
        self.peak_working_estimate = max(self.peak_working_estimate, working_bytes)
        self.records.append(dict(layer=layer, probabilities=item, span_values=values))
        self.observed.append(layer)
