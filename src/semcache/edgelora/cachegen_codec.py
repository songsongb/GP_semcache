"""CacheGen-style codec adapted to one LoRA Q/K/V projection delta.

The original CacheGen codec targets KV cache tensors.  Here a PEFT LoRA
projection delta [B,T,D] is treated as a one-layer CacheGen tensor [1,B,T,D].
CDFs can be built offline and reused at runtime.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch

LMCACHE_ROOT = Path(
    "/data/qwe1234k/repos/kvcache/cachegen_qwen3/vendor/CacheGen/LMCache"
)
if str(LMCACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(LMCACHE_ROOT))

import torchac_cuda
from lmcache.storage_backend.serde.cachegen_encoder import (
    torch_quant_vectorized,
    encode_ntokens,
)
from lmcache.storage_backend.serde.cachegen_decoder import (
    do_dequantize,
    decode_chunk,
)
from lmcache.storage_backend.serde.cachegen_basics import CacheGenGPUBytestream
import lmcache.storage_backend.serde.cachegen_basics as CGBasics


def _nbytes(x: torch.Tensor) -> int:
    return int(x.numel() * x.element_size())


def _bins_tensor(num_bins: int, device: torch.device) -> torch.Tensor:
    if not isinstance(num_bins, int) or num_bins < 2:
        raise ValueError("num_bins must be an integer >= 2")
    # CacheGen expects one bin count per layer.  We map one LoRA projection to L=1.
    return torch.tensor([num_bins], dtype=torch.int32, device=device)


def _prepare(x: torch.Tensor):
    if x.ndim != 3 or x.shape[0] != 1 or x.numel() == 0:
        raise ValueError("LoRA delta must be nonempty [1, T, D]")
    # CacheGen single-layer representation: [L=1, T, D]
    return x[0].unsqueeze(0).contiguous().cuda()


def build_lora_cdf(x: torch.Tensor, num_bins: int = 32) -> torch.Tensor:
    """Build a CacheGen arithmetic-coding CDF for one LoRA delta tensor."""
    fp = _prepare(x)
    bins = _bins_tensor(num_bins, fp.device)
    quantized, _ = torch_quant_vectorized(bins, fp)
    return torchac_cuda.calculate_cdf(quantized, int(bins.max().item()))


def encode_lora_delta(
    x: torch.Tensor,
    num_bins: int = 32,
    cdf: Optional[torch.Tensor] = None,
) -> Dict:
    """Quantize + arithmetic-code a [1,T,D] LoRA projection delta.

    If ``cdf`` is supplied it is reused (offline/pre-shared profile).  The
    returned ``sizes`` separates one-time CDF bytes from runtime payload bytes.
    """
    fp = _prepare(x)
    _, T, D = fp.shape
    bins = _bins_tensor(num_bins, fp.device)
    quantized, max_tensors = torch_quant_vectorized(bins, fp)
    if cdf is None:
        cdf = torchac_cuda.calculate_cdf(quantized, int(bins.max().item()))
    else:
        cdf = cdf.to(fp.device)

    max_chunk = CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK
    output_buffer = torch.zeros((1, D, max_chunk), dtype=torch.uint8, device=fp.device)
    output_lengths = torch.zeros((1, D), dtype=torch.int32, device=fp.device)
    chunks = []

    for start in range(0, T, max_chunk):
        end = min(start + max_chunk, T)
        output_lengths.zero_()
        bytestream = encode_ntokens(
            cdf,
            quantized[:, start:end, :],
            output_buffer,
            output_lengths,
        )
        chunks.append(
            CacheGenGPUBytestream(
                bytestream=bytestream,
                bytestream_lengths=output_lengths.clone(),
                ntokens=end - start,
            )
        )

    stream_bytes = sum(_nbytes(c.bytestream) for c in chunks)
    length_bytes = sum(_nbytes(c.bytestream_lengths) for c in chunks)
    cdf_bytes = _nbytes(cdf)
    max_bytes = _nbytes(max_tensors)
    bins_bytes = _nbytes(bins)
    # Shape/token metadata is tiny but count explicit integers for honest accounting.
    metadata_bytes = max_bytes + length_bytes + bins_bytes
    runtime_total_bytes = stream_bytes + metadata_bytes
    total_bytes_with_cdf = runtime_total_bytes + cdf_bytes

    return {
        "shape": tuple(x.shape),
        "dtype": x.dtype,
        "device": x.device,
        "bins": bins,
        "num_bins": int(num_bins),
        "cdf": cdf,
        "max_tensors": max_tensors,
        "chunks": chunks,
        "sizes": {
            "raw_bytes": _nbytes(x),
            "stream_bytes": stream_bytes,
            "cdf_bytes": cdf_bytes,
            "max_bytes": max_bytes,
            "length_bytes": length_bytes,
            "bins_bytes": bins_bytes,
            "metadata_bytes": metadata_bytes,
            "runtime_total_bytes": runtime_total_bytes,
            "total_bytes_with_cdf": total_bytes_with_cdf,
        },
    }


def decode_lora_delta(encoded: Dict) -> torch.Tensor:
    """Decode one object returned by :func:`encode_lora_delta`."""
    B, T, D = encoded["shape"]
    if B != 1:
        raise ValueError("Only batch size 1 is supported")
    device = encoded["cdf"].device
    quantized_output = torch.zeros((1, T, D), dtype=torch.uint8, device=device)
    start = 0
    for chunk in encoded["chunks"]:
        end = start + int(chunk.ntokens)
        decode_chunk(encoded["cdf"], chunk, quantized_output[:, start:end, :])
        start = end
    if start != T:
        raise RuntimeError("Decoded token count does not match encoded shape")
    decoded = do_dequantize(
        quantized_output.float(),
        encoded["bins"],
        encoded["max_tensors"],
    )
    return decoded.to(dtype=encoded["dtype"]).reshape(B, T, D)
