"""Logical QKV payload only, excluding metadata/packing/allocator overhead."""
import math
from .cost_model import positive


def qkv_block_bytes(tokens, layers, q_dimension, kv_dimension, precision_bits):
    for name, v in [('tokens', tokens), ('layers', layers), ('q_dimension', q_dimension),
                    ('kv_dimension', kv_dimension), ('precision_bits', precision_bits)]:
        positive(v, name)
        if not isinstance(v, int):
            raise ValueError(f'{name} must be integer')
    # Packed payload rounded to whole bytes. INT4 weights do not imply INT4 QKV.
    return math.ceil(layers*tokens*(q_dimension+2*kv_dimension)*precision_bits/8)


def qkv_logical_bytes(tokens, layers, hidden_size, bytes_per_element):
    positive(bytes_per_element, 'bytes_per_element')
    bits = bytes_per_element*8
    if int(bits) != bits:
        raise ValueError('Precision must contain whole bits')
    return qkv_block_bytes(tokens, layers, hidden_size, hidden_size, int(bits))
