"""Non-invasive raw OPT projections; one record per layer per forward."""
from contextlib import contextmanager
from types import SimpleNamespace


@contextmanager
def qkv_capture(model, layers=None, storage_device='cpu', validate=True):
    import torch
    from .model_adapter import OPTModelAdapter
    from semcache.evaluation.qkv_metrics import similarity
    adapter = OPTModelAdapter(model)
    selected = list(range(len(adapter.layers))) if layers is None else list(layers)
    if len(set(selected)) != len(selected) or any(i < 0 or i >= len(adapter.layers) for i in selected):
        raise ValueError('Invalid capture layers')
    capture = SimpleNamespace(records={})
    handles = []
    def hook(layer_idx, name):
        def record(module, args, output):
            record = capture.records.setdefault(layer_idx, {'layer_idx': layer_idx, 'validation': {}})
            if name in record:
                raise RuntimeError('Use a new capture context for each forward')
            # Direct forward avoids recursively invoking this hook. It uses the
            # exact loaded module on its actual input, before scaling/reshaping.
            if validate:
                with torch.inference_mode():
                    independent = module.forward(args[0])
                torch.testing.assert_close(independent, output)
                record['validation'][name] = similarity(output, independent)
            if name == 'q':
                record['hidden_states'] = args[0].detach().to(storage_device or output.device).clone()
            record[name] = output.detach().to(storage_device or output.device).clone()
            record.update(token_count=output.shape[1], shape=list(output.shape),
                          dtype=str(output.dtype), device=str(output.device),
                          storage_device=str(record[name].device))
            record['metadata'] = adapter.projection_metadata(layer_idx, output)
        return record
    try:
        for i in selected:
            for name in ('q', 'k', 'v'):
                handles.append(adapter.projection_modules(i)[name].register_forward_hook(hook(i, name)))
        yield capture
        if set(capture.records) != set(selected) or any(not all(n in r for n in ('q','k','v')) for r in capture.records.values()):
            raise RuntimeError('Forward did not execute all selected projections')
    finally:
        for handle in handles:
            handle.remove()
