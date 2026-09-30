"""CacheGen-reconstructed EdgeLoRA projection path.

Logical path: ES base projection + UD LoRA delta -> CacheGen encode/decode -> ES
recombination.  This is single-process emulation; it is not a physical network.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import time

from semcache.models.lora_decomposition import projection_parts
from semcache.edgelora.cachegen_codec import encode_lora_delta, decode_lora_delta


def _record_sizes(sizes):
    return {
        "raw_bytes": int(sizes["raw_bytes"]),
        "stream_bytes": int(sizes["stream_bytes"]),
        "cdf_bytes": int(sizes["cdf_bytes"]),
        "max_bytes": int(sizes["max_bytes"]),
        "length_bytes": int(sizes["length_bytes"]),
        "bins_bytes": int(sizes["bins_bytes"]),
        "metadata_bytes": int(sizes["metadata_bytes"]),
        "runtime_total_bytes": int(sizes["runtime_total_bytes"]),
        "total_bytes_with_cdf": int(sizes["total_bytes_with_cdf"]),
    }


@contextmanager
def cachegen_reconstructed_projection_path(
    adapter,
    adapter_name,
    layers=None,
    num_bins=32,
    cdf_map=None,
    num_bins_map=None,
    allow_multiple_forwards=False,
    measure_codec_latency=False,
):
    """Replace PEFT q/k/v outputs with base + decoded compressed LoRA delta.

    ``cdf_map[(layer, name)]`` can hold pre-shared CDFs.  ``num_bins_map``
    enables layer-wise schedules while preserving the scalar ``num_bins`` API.
    """
    import torch

    selected = list(range(len(adapter.layers))) if layers is None else list(layers)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select unique nonempty layers")

    audit = SimpleNamespace(records=[], seen=set())
    handles = []

    def hook(layer, name):
        def replace(module, args, output):
            key = (layer, name)
            if not allow_multiple_forwards and key in audit.seen:
                raise RuntimeError("Use one forward per reconstruction context")

            base, delta, _ = projection_parts(module, args[0], adapter_name)
            shared_cdf = None if cdf_map is None else cdf_map[key]
            current_num_bins = num_bins if num_bins_map is None else int(num_bins_map[key])

            encode_ms = decode_ms = 0.0
            if measure_codec_latency and delta.is_cuda:
                torch.cuda.synchronize(delta.device)
            t0 = time.perf_counter()
            encoded = encode_lora_delta(delta, num_bins=current_num_bins, cdf=shared_cdf)
            if measure_codec_latency and delta.is_cuda:
                torch.cuda.synchronize(delta.device)
            t1 = time.perf_counter()

            decoded = decode_lora_delta(encoded)
            if measure_codec_latency and delta.is_cuda:
                torch.cuda.synchronize(delta.device)
            t2 = time.perf_counter()

            if measure_codec_latency:
                encode_ms = (t1 - t0) * 1000.0
                decode_ms = (t2 - t1) * 1000.0

            reconstructed = (base + decoded.to(base)).to(base.dtype)
            record = {
                "layer": int(layer),
                "tensor_type": name,
                "num_bins": current_num_bins,
                **_record_sizes(encoded["sizes"]),
                "encode_ms": float(encode_ms),
                "decode_ms": float(decode_ms),
                "codec_ms": float(encode_ms + decode_ms),
            }
            audit.records.append(record)
            audit.seen.add(key)
            return reconstructed
        return replace

    try:
        for layer in selected:
            for name, module in adapter.projection_modules(layer).items():
                handles.append(module.register_forward_hook(hook(layer, name)))
        yield audit
        if not allow_multiple_forwards and len(audit.seen) != 3 * len(selected):
            raise RuntimeError("Forward did not execute all reconstructed projections")
    finally:
        for handle in handles:
            handle.remove()
