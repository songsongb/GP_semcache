"""Single-forward substitution of base_raw_unscaled_linear_projection outputs."""
from contextlib import contextmanager
from types import SimpleNamespace


@contextmanager
def qkv_injection(adapter, layer, cached_qkv, source_window, target_window, mode='qkv'):
    """Windows are end-exclusive; source coordinates index the supplied payload.

    Audit all three projections against their fresh outputs in this forward.
    This is correctness instrumentation, not a compute-saving implementation.
    """
    import torch
    modules = adapter.projection_modules(layer)
    if mode not in ('q', 'k', 'v', 'qkv'):
        raise ValueError('Invalid injection mode')
    if len(cached_qkv) != 3:
        raise ValueError('Need three cached projections')
    ss, se = source_window
    ts, te = target_window
    if any(type(i) is not int for i in (ss, se, ts, te)) or ss < 0 or ts < 0 or se <= ss or te <= ts or se-ss != te-ts:
        raise ValueError('Invalid or mismatched source/target windows')
    shape = cached_qkv[0].shape
    if len(shape) != 3 or shape[0] != 1 or shape[2] <= 0 or se > shape[1] or any(t.shape != shape for t in cached_qkv):
        raise ValueError('Expected equal batch-one [batch, tokens, hidden] cached tensors')
    sources = {n: t.detach()[:, ss:se, :].clone() for n, t in zip('qkv', cached_qkv)}
    audit = SimpleNamespace(records={}, injected_tensor_bytes=0)
    handles = []

    def hook(name):
        def replace(module, args, output):
            if name in audit.records:
                raise RuntimeError('Use a new injection context for each forward')
            if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != shape[2] or te > output.shape[1]:
                raise ValueError('Destination projection shape/window mismatch')
            with torch.inference_mode():
                fresh = output.detach()
                result = fresh
                if name in mode:
                    source = sources[name].to(device=output.device, dtype=output.dtype)
                    result = fresh.clone()
                    result[:, ts:te, :] = source
                    if not torch.equal(result[:, ts:te, :], source):
                        raise AssertionError('Injected slice differs from retrieved source')
                    audit.injected_tensor_bytes += source.numel()*source.element_size()
                if not torch.equal(result[:, :ts], fresh[:, :ts]) or not torch.equal(result[:, te:], fresh[:, te:]):
                    raise AssertionError('Non-target projection positions changed')
                if name not in mode and not torch.equal(result, fresh):
                    raise AssertionError('Unselected projection changed')
                audit.records[name] = True
                return result
        return replace

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(hook(name)))
        yield audit
        if set(audit.records) != set('qkv'):
            raise RuntimeError('Forward did not execute all projections')
    finally:
        for handle in handles:
            handle.remove()
