"""Single-process ES/UD projection emulation; no network or latency claims."""
from contextlib import contextmanager
from semcache.models.lora_decomposition import decompose_projection, assert_decomposition


def communication_accounting(hidden_states, q_lora, k_lora, v_lora):
    tensors = (hidden_states, q_lora, k_lora, v_lora)
    if hidden_states.ndim != 3 or hidden_states.numel() == 0 or any(t.shape != hidden_states.shape for t in tensors):
        raise ValueError('OPT exchange requires equal nonempty [batch, seq, hidden] tensors')
    sizes = [t.numel() * t.element_size() for t in tensors]
    upstream = hidden_states.numel()
    downstream = sum(t.numel() for t in tensors[1:])
    b, n, d = hidden_states.shape
    return dict(batch_size=b, seq_len=n, hidden_dim=d, es_to_ud_elements=upstream,
        ud_to_es_elements=downstream, total_comm_elements=upstream+downstream,
        paper_comm_elements=4*n*d, paper_comm_elements_scope='per sequence (batch one)',
        measured_tensor_bytes=sum(sizes), hidden_tensor_bytes=sizes[0],
        q_lora_bytes=sizes[1], k_lora_bytes=sizes[2], v_lora_bytes=sizes[3])


def edge_projection(hidden_states, projection_module, adapter_name, tolerance=1e-6):
    """ES computes base, UD computes delta on the same device; return explicit parts."""
    parts = decompose_projection(projection_module, hidden_states, adapter_name)
    assert_decomposition(parts, tolerance)
    return parts


def edge_qkv_projection(hidden_states, adapter, layer, adapter_name, tolerance=1e-6):
    parts = {n: edge_projection(hidden_states, m, adapter_name, tolerance)
             for n, m in adapter.projection_modules(layer).items()}
    comm = communication_accounting(hidden_states, *(parts[n].lora_delta for n in 'qkv'))
    return parts, comm


@contextmanager
def reconstructed_projection_path(adapter, adapter_name, layers=None, tolerance=1e-6):
    """Replace native raw totals with independent ES+UD totals before attention.

    Fresh native outputs are still computed. This validates correctness, not speed.
    """
    selected = list(range(len(adapter.layers))) if layers is None else list(layers)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Select unique nonempty layers')
    handles, seen = [], set()
    def hook(layer, name):
        def replace(module, args, output):
            from semcache.models.lora_decomposition import ProjectionComponents, projection_parts
            key = (layer, name)
            if key in seen:
                raise RuntimeError('Use one forward per reconstruction context')
            base, delta, combined = projection_parts(module, args[0], adapter_name)
            assert_decomposition(ProjectionComponents(base, delta, combined, output), tolerance)
            seen.add(key)
            return combined
        return replace
    try:
        for layer in selected:
            for name, module in adapter.projection_modules(layer).items():
                handles.append(module.register_forward_hook(hook(layer, name)))
        yield seen
        if len(seen) != 3*len(selected):
            raise RuntimeError('Forward did not execute all reconstructed projections')
    finally:
        for handle in handles:
            handle.remove()
