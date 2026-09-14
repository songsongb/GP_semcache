"""Single-forward, batch-one native PEFT subset projection; no overwrite path."""
from contextlib import contextmanager
from types import SimpleNamespace
from semcache.models.lora_decomposition import validate_projection


@contextmanager
def mixed_projection_path(adapter, adapter_name, hits, sequence_length):
    """Patch instance forwards only; restore exact original attributes in finally.

    Module hooks must be absent: M2 validation hooks independently re-project
    full inputs and would invalidate work accounting. Captures here are passive.
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
    audit = SimpleNamespace(records={}, projections={}, reused_mask=mask)
    originals = []

    def replacement(layer, name, module, native):
        def forward(hidden_states, *args, **kwargs):
            key = (layer, name)
            if key in audit.records:
                raise RuntimeError('Use one forward per mixed context')
            if args or kwargs or hidden_states.shape != (1, sequence_length, module.in_features):
                raise ValueError('Mixed path requires batch-one prefill hidden states only')
            with torch.inference_mode():
                # This is the ONLY invocation of the native PEFT forward.
                index = fresh_positions.to(hidden_states.device)
                fresh = native(hidden_states.index_select(1, index)) if len(index) else None
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
                    native_projection_rows=len(index), reused_projection_rows=reused,
                    expected_saved_projection_positions=reused, native_call_count=int(fresh is not None),
                    native_positions=fresh_positions.tolist(), component_scope='total_qkv')
                audit.projections.setdefault(layer, {})[name] = output.detach().cpu().clone()
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
