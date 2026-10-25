"""Shared M9-A.2 attention arithmetic, without history or query identity.

Extracted without changing arithmetic order, masks, finite checks or counters.
Diagnostics supply Attribution; production supplies a disabled stage recorder.
"""
from collections import defaultdict
from contextlib import nullcontext
import math
from .attention_impact import PaperRowL2SumReducer

MODES = ('CURRENT_BLOCKWISE', 'TOKEN_PRECOMPUTE', 'TOKEN_PREFIX_SUM')
PARTS = ('impact_attention_access_ms', 'impact_row_l2_reduction_ms',
         'impact_device_to_host_ms', 'impact_window_aggregation_ms', 'impact_finalize_ms')
EXECUTIONS = ('legacy_blockwise', 'token_precompute', 'token_prefix_sum')

class NoAttribution:
    def __init__(self):
        self.counts = defaultdict(int)

    def stage(self, field, device_work=False):
        return nullcontext()


def _validate(attention, n):
    if (attention is None or attention.ndim != 4 or attention.shape[0] != 1
            or attention.shape[1] < 1 or attention.shape[2] != attention.shape[3]
            or attention.shape[2] != n):
        raise ValueError('Expected batch-one square prefill attention')


def _mask(torch, mask, n, device):
    if mask is None:
        return None
    mask = mask.to(device=device, dtype=torch.bool)
    if mask.ndim == 2 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 1 or len(mask) != n:
        raise ValueError('Valid mask must cover prefill positions')
    return mask


def compute_window_values(attentions, windows, mode, valid_attention_mask=None, *,
                          timer=None, production_reference=False):
    """Return values in candidate order; never group keys or append history."""
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
    timer = timer if timer is not None else NoAttribution()
    values_out = []
    if production_reference:
        reducer = PaperRowL2SumReducer()
        for w in windows:
            mask = valid_attention_mask if valid_attention_mask is not None else torch.ones(n, dtype=torch.bool)
            values_out.append(reducer.reduce(attentions, w.start, w.end, mask))
    elif mode == MODES[0]:
        for w in windows:
            total = 0.
            with timer.stage(PARTS[0]):
                # The engine currently constructs a CPU mask for each candidate.
                mask_input = valid_attention_mask if valid_attention_mask is not None else torch.ones(n, dtype=torch.bool)
                timer.counts['cpu_mask_constructions'] += valid_attention_mask is None
            for attention in attentions:
                with timer.stage(PARTS[0], True):
                    _validate(attention, n)
                    tensor = attention.detach().double()[0, :, w.start:w.end, :]
                    timer.counts['layer_visits'] += 1
                    timer.counts['full_attention_double_calls'] += 1
                    timer.counts['full_attention_conversion_allocations'] += attention.dtype != torch.float64
                    timer.counts['full_attention_elements_visited'] += attention.numel()
                    timer.counts['slice_views'] += 1
                with timer.stage(PARTS[1], True):
                    valid = (torch.arange(n, device=device)[None, :] <=
                             torch.arange(w.start, w.end, device=device)[:, None])
                    mask = _mask(torch, mask_input, n, device)
                    valid &= mask[w.start:w.end, None] & mask[None, :]
                    tensor = tensor.masked_fill(~valid.unsqueeze(0), 0)
                    finite = torch.isfinite(tensor).all()
                    timer.counts['mask_to_device_calls'] += 1
                    timer.counts['blocking_mask_h2d_calls'] += str(device).startswith('cuda') and not mask_input.is_cuda
                    timer.counts['masked_tensor_copies'] += 1
                with timer.stage(PARTS[2]):
                    ok = bool(finite)
                    timer.counts['finite_scalar_host_reads'] += 1
                if not ok:
                    raise ValueError('Nonfinite valid attention')
                with timer.stage(PARTS[1], True):
                    rows = tensor.norm(dim=-1).mean(dim=0)
                    timer.counts['row_l2_calls'] += 1
                    timer.counts['row_l2_head_rows'] += attention.shape[1]*(w.end-w.start)
                    timer.counts['head_reduction_calls'] += 1
                with timer.stage(PARTS[3], True):
                    value = rows.sum()
                with timer.stage(PARTS[2]):
                    total += value.item()
                    timer.counts['impact_scalar_host_reads'] += 1
            if not math.isfinite(total):
                raise ValueError('Nonfinite attention impact')
            values_out.append(total)
    elif windows:
        with timer.stage(PARTS[0], True):
            mask_input = valid_attention_mask if valid_attention_mask is not None else torch.ones(n, dtype=torch.bool)
            timer.counts['cpu_mask_constructions'] += valid_attention_mask is None
            mask = _mask(torch, mask_input, n, device)
            timer.counts['mask_to_device_calls'] += 1
            timer.counts['blocking_mask_h2d_calls'] += str(device).startswith('cuda') and not mask_input.is_cuda
            timer.counts['window_index_h2d_constructions'] += 2 if str(device).startswith('cuda') else 0
            starts = torch.tensor([w.start for w in windows], device=device)
            ends = torch.tensor([w.end for w in windows], device=device)
            positions = torch.arange(n, device=device)
            membership = (positions[None, :] >= starts[:, None]) & (positions[None, :] < ends[:, None])
            valid = positions[None, :] <= positions[:, None]
            valid &= mask[:, None] & mask[None, :]
            # Do not reject nonfinite values outside the queried rows: production
            # only inspects rows belonging to at least one candidate window.
            valid &= membership.any(dim=0)[:, None]
            tokens = torch.zeros(n, device=device, dtype=torch.float64)
            finite_flags = []
        for attention in attentions:
            with timer.stage(PARTS[0], True):
                _validate(attention, n)
                if attention.device != device:
                    raise ValueError('Diagnostic requires a single attention device')
                tensor = attention.detach().double()[0]
                timer.counts['layer_visits'] += 1
                timer.counts['full_attention_double_calls'] += 1
                timer.counts['full_attention_conversion_allocations'] += attention.dtype != torch.float64
                timer.counts['full_attention_elements_visited'] += attention.numel()
                timer.counts['slice_views'] += 1
            with timer.stage(PARTS[1], True):
                tensor = tensor.masked_fill(~valid.unsqueeze(0), 0)
                finite_flags.append(torch.isfinite(tensor).all())
                tokens = tokens + tensor.norm(dim=-1).mean(dim=0)
                timer.counts['masked_tensor_copies'] += 1
                timer.counts['row_l2_calls'] += 1
                timer.counts['row_l2_head_rows'] += attention.shape[1]*n
                timer.counts['head_reduction_calls'] += 1
        with timer.stage(PARTS[3], True):
            if mode == 'TOKEN_PRECOMPUTE':
                impacts = torch.where(membership, tokens[None, :], 0.).sum(dim=1)
            else:
                # Fixed-order inclusive scan uses deterministic elementwise ops,
                # preserving the fixture's deterministic-algorithms setting even
                # on backends without a deterministic floating-point cumsum.
                prefix = tokens
                offset = 1
                while offset < n:
                    prefix = torch.cat((prefix[:offset], prefix[offset:] + prefix[:-offset]))
                    offset *= 2
                    timer.counts['prefix_scan_steps'] += 1
                prefix = torch.cat((tokens.new_zeros(1), prefix))
                impacts = prefix[ends] - prefix[starts]
            # One transfer includes the finite flag; no per-layer/per-block item.
            packed = torch.cat((torch.stack(finite_flags).all().double().reshape(1), impacts))
        with timer.stage(PARTS[2]):
            host = packed.cpu().tolist()
            timer.counts['batched_host_materializations'] += 1
        if not host[0] or not all(math.isfinite(v) for v in host[1:]):
            raise ValueError('Nonfinite valid attention/impact')
        values_out = host[1:]
    return values_out
