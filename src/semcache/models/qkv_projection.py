"""Tensor helpers use [batch, token, hidden] before head splitting."""

def extract_qkv_positions(q, k, v, positions):
    import torch
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Expected equal [batch, token, hidden] shapes")
    positions = list(positions)
    if any(p < 0 or p >= q.shape[1] for p in positions):
        raise ValueError("Position out of range")
    with torch.inference_mode():
        return tuple(t.index_select(1, torch.tensor(positions, dtype=torch.long, device=t.device)) for t in (q, k, v))


def merge_cached_and_new_qkv(blocks, sequence_length):
    """Blocks are (destination_positions, (q,k,v)); reject overlaps and gaps.

    A permutation test validates placement, not contextual equivalence of blocks.
    """
    import torch
    blocks = [(list(pos), qkv) for pos, qkv in blocks]
    positions = [p for pos, _ in blocks for p in pos]
    if not blocks or sorted(positions) != list(range(sequence_length)):
        raise ValueError("Blocks must cover each destination exactly once")
    ref = blocks[0][1][0]
    with torch.inference_mode():
        outputs = tuple(ref.new_empty((ref.shape[0], sequence_length, ref.shape[2])) for _ in range(3))
        for pos, qkv in blocks:
            if len(qkv) != 3 or any(t.shape != (ref.shape[0], len(pos), ref.shape[2]) or t.device != ref.device or t.dtype != ref.dtype for t in qkv):
                raise ValueError("Incompatible QKV block")
            index = torch.tensor(pos, dtype=torch.long, device=ref.device)
            for dst, src in zip(outputs, qkv):
                dst.index_copy_(1, index, src)
    return outputs
