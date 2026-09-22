"""Logical single-user budget; allocates no tensors or capacity-sized arrays."""
from .common import integer

UD_CAPACITY = 8 * 1024**3


def ud_memory(dims, sequence_length, *, parameter_bytes=4, temporary_element_bytes=4,
              user_local_cache_bytes=0, capacity_bytes=UD_CAPACITY):
    for name, value, minimum in [('sequence_length', sequence_length, 1),
            ('parameter_bytes', parameter_bytes, 1), ('temporary_element_bytes', temporary_element_bytes, 1),
            ('user_local_cache_bytes', user_local_cache_bytes, 0), ('capacity_bytes', capacity_bytes, 1)]:
        integer(value, name, minimum)
    # All-layer Q/K/V A and B parameters; no full base model or 8 GiB allocation.
    adapters = 6*dims.layers*dims.hidden_size*dims.rank*parameter_bytes
    inputs = sequence_length*dims.hidden_size*temporary_element_bytes
    outputs = 3*inputs
    intermediates = 3*sequence_length*dims.rank*temporary_element_bytes
    total = adapters + inputs + outputs + intermediates + user_local_cache_bytes
    return dict(adapter_parameter_bytes=adapters, input_temporary_bytes=inputs,
        output_temporary_bytes=outputs, low_rank_temporary_bytes=intermediates,
        user_local_cache_bytes=user_local_cache_bytes, estimated_usage_bytes=total,
        ud_memory_capacity_bytes=capacity_bytes, fits=total <= capacity_bytes,
        provenance='SIMULATED', capacity_provenance='PAPER_REFERENCE' if capacity_bytes == UD_CAPACITY else 'SIMULATED',
        capacity_interpretation='8 GiB (8*1024**3) logical cap',
        scope='QKV adapters plus one-layer working set; not whole-UD peak RSS',
        excluded='embedding/output weight tables, logits, runtime, allocator, tokenizer, OS, wire cast buffers')
