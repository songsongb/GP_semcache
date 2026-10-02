"""Single-forward, batch-one native PEFT subset projection; no overwrite path."""
from contextlib import contextmanager
from types import SimpleNamespace
from semcache.models.lora_decomposition import validate_projection
from semcache.metrics.timing import resolve_cuda_event_pairs


@contextmanager
def mixed_projection_path(adapter, adapter_name, hits, sequence_length, *, measure_cuda=False,
                          fresh_projection=None):
    """Patch instance forwards only; restore exact original attributes in finally.

    Module hooks must be absent: M2 validation hooks independently re-project
    full inputs and would invalidate work accounting. Captures here are passive.
    ``fresh_projection`` optionally reconstructs fresh rows (e.g. transported
    LoRA deltas). Cached total projections bypass that callback entirely.
    """
    import torch
    if sequence_length < 1:
        raise ValueError('Nonempty sequence required')
    mask = torch.zeros(sequence_length, dtype=torch.bool)
    for hit in hits:
        w, entry = hit.window, hit.entry
        if (not 0 <= w.start < w.end <= sequence_length or mask[w.start:w.end].any()
                or entry.qkv_metadata.get('component_scope') != 'total_qkv'
                or w.token_ids != entry.token_ids or not entry.tensors):
            raise ValueError('Invalid, overlapping or non-total cached span')
        mask[w.start:w.end] = True
    fresh_positions = (~mask).nonzero().flatten()
    reused = int(mask.sum())
    assert reused + len(fresh_positions) == sequence_length
    audit = SimpleNamespace(records={}, projections={}, reused_mask=mask, cuda_event_pairs=[])
    originals = []

    def replacement(layer, name, module, native):
        def forward(hidden_states, *args, **kwargs):
            key = (layer, name)
            if key in audit.records:
                raise RuntimeError('Use one forward per mixed context')
            if args or kwargs or hidden_states.shape != (1, sequence_length, module.in_features):
                raise ValueError('Mixed path requires batch-one prefill hidden states only')
            start = end = None
            if measure_cuda and hidden_states.is_cuda:
                start, end = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                with torch.cuda.device(hidden_states.device):
                    start.record()
            with torch.inference_mode():
                # This is the ONLY invocation of the native PEFT forward.
                index = fresh_positions.to(hidden_states.device)
                fresh_inputs = hidden_states.index_select(1, index)
                fresh = ((native(fresh_inputs) if fresh_projection is None else
                          fresh_projection(layer, name, module, fresh_inputs, native))
                         if len(index) else None)
                dtype = fresh.dtype if fresh is not None else module.get_base_layer().weight.dtype
                output = torch.empty((1, sequence_length, module.out_features), device=hidden_states.device, dtype=dtype)
                if fresh is not None:
                    output.index_copy_(1, index, fresh)
                for hit in hits:
                    source = hit.entry.tensors[layer]['qkv'.index(name)]
                    w = hit.window
                    if source.shape != (1, w.end-w.start, module.out_features):
                        raise ValueError('Cached projection shape mismatch')
                    output[:, w.start:w.end] = source.to(output)
                audit.records[key] = dict(layer=layer, tensor_type=name, full_sequence_length=sequence_length,
                    native_projection_rows=len(index) if fresh_projection is None else 0,
                    reconstructed_projection_rows=len(index) if fresh_projection is not None else 0,
                    reused_projection_rows=reused,
                    expected_saved_projection_positions=reused,
                    native_call_count=int(fresh is not None and fresh_projection is None),
                    native_positions=fresh_positions.tolist(), component_scope='total_qkv')
                # Keep passive detached captures on the execution device. Moving every
                # projection to CPU here creates a host barrier per Q/K/V call;
                # admitted spans are copied to storage later by CacheEntry.
                audit.projections.setdefault(layer, {})[name] = output.detach()
                if end is not None:
                    with torch.cuda.device(hidden_states.device):
                        end.record()
                    audit.cuda_event_pairs.append((start, end))
                return output
        return forward

    try:
        for layer in range(len(adapter.layers)):
            for name, module in adapter.projection_modules(layer).items():
                validate_projection(module, adapter_name)
                if module._forward_hooks or module._forward_pre_hooks or 'forward' in module.__dict__:
                    raise ValueError('Mixed path requires uninstrumented projection modules')
                for hit in hits:
                    if layer not in hit.entry.tensors:
                        raise ValueError('Cached total QKV must cover all projection layers')
                originals.append(module)
                module.forward = replacement(layer, name, module, module.forward)
        yield audit
        if len(audit.records) != 3*len(adapter.layers):
            raise RuntimeError('Forward did not execute all mixed projections')
    finally:
        for module in originals:
            del module.forward


def mixed_qkv_elapsed_ms(audit, *, enclosing_region_synchronized=False):
    """Sum disjoint projection events; an enclosing event normally owns the sync."""
    if not audit.cuda_event_pairs:
        return None
    return sum(resolve_cuda_event_pairs(audit.cuda_event_pairs,
        synchronize=not enclosing_region_synchronized))
