def qkv_logical_bytes(tokens, layers, hidden_size, bytes_per_element):
    if min(tokens, layers, hidden_size, bytes_per_element) < 1:
        raise ValueError("QKV dimensions must be positive")
    return 3 * tokens * layers * hidden_size * bytes_per_element
