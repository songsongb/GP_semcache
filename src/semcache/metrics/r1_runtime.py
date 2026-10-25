"""Opt-in R1 timing only; legacy engine and mixed executor are never edited.

Single-threaded runner: scoped bindings restore the engine module on exit.
The detailed executor follows mixed_projection_path's gather/native/scatter
algorithm. Minimal runs use the original executor, with no new inner events.
"""
from contextlib import contextmanager, nullcontext
from collections import defaultdict
from time import perf_counter_ns
from types import SimpleNamespace
from unittest.mock import patch

from .r1 import audit_projection_records


def synchronize(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class Intervals:
    """Host submission times and nested CUDA-stream intervals, resolved later."""
    def __init__(self, device):
        self.device = device
        self.cpu = defaultdict(float)
        self.events = []
        self.h2d_cache_bytes = 0
        self.cache_fetch_calls = 0

    @contextmanager
    def span(self, name):
        import torch
        start_ns = perf_counter_ns()
        pair = None
        if str(self.device).startswith("cuda"):
            with torch.cuda.device(self.device):
                pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                pair[0].record()
        try:
            yield
        finally:
            if pair:
                with torch.cuda.device(self.device):
                    pair[1].record()
                self.events.append((name, pair))
            self.cpu[name] += (perf_counter_ns() - start_ns) / 1e6

    def export(self):
        # Request end has already synchronized. No per-subspan synchronizations.
        gpu = defaultdict(float)
        for name, (start, end) in self.events:
            gpu[name] += start.elapsed_time(end)
        names = ("native_qkv", "cache_fetch", "tensor_assembly", "index_transfer", "mixed_executor")
        values = {name + "_host_ms": self.cpu.get(name, 0.) for name in names}
        values.update({name + "_gpu_ms": gpu.get(name, 0.) if str(self.device).startswith("cuda") else None for name in names})
        return values


@contextmanager
def detailed_mixed(adapter, adapter_name, hits, sequence_length, *, book, measure_cuda=False):
    """Same TOTAL-QKV subset execution as the legacy executor, with subspans."""
    import torch
    from semcache.models.lora_decomposition import validate_projection
    if sequence_length < 1:
        raise ValueError("Nonempty sequence required")
    mask = torch.zeros(sequence_length, dtype=torch.bool)
    for hit in hits:
        w, entry = hit.window, hit.entry
        if (not 0 <= w.start < w.end <= sequence_length or mask[w.start:w.end].any()
                or entry.qkv_metadata.get("component_scope") != "total_qkv"
                or w.token_ids != entry.token_ids or not entry.tensors):
            raise ValueError("Invalid cached span")
        mask[w.start:w.end] = True
    fresh_positions = (~mask).nonzero().flatten()
    reused = int(mask.sum())
    audit = SimpleNamespace(records={}, projections={}, reused_mask=mask, cuda_event_pairs=[])
    originals = []

    def replacement(layer, name, module, native):
        def forward(hidden_states, *args, **kwargs):
            key = (layer, name)
            if key in audit.records or args or kwargs or hidden_states.shape != (1, sequence_length, module.in_features):
                raise ValueError("Expected one batch-one prefill per mixed context")
            with book.span("mixed_executor"), torch.inference_mode():
                with book.span("index_transfer"):
                    index = fresh_positions.to(hidden_states.device)
                with book.span("tensor_assembly"):
                    fresh_inputs = hidden_states.index_select(1, index)
                fresh = None
                if len(index):
                    with book.span("native_qkv"):
                        fresh = native(fresh_inputs)
                dtype = fresh.dtype if fresh is not None else module.get_base_layer().weight.dtype
                with book.span("tensor_assembly"):
                    output = torch.empty((1, sequence_length, module.out_features), device=hidden_states.device, dtype=dtype)
                    if fresh is not None:
                        output.index_copy_(1, index, fresh)
                for hit in hits:
                    source = hit.entry.tensors[layer]["qkv".index(name)]
                    w = hit.window
                    if source.shape != (1, w.end-w.start, module.out_features):
                        raise ValueError("Cached projection shape mismatch")
                    book.cache_fetch_calls += 1
                    if source.device.type == "cpu" and output.is_cuda:
                        book.h2d_cache_bytes += source.numel() * source.element_size()
                    with book.span("cache_fetch"):
                        fetched = source.to(output)
                    with book.span("tensor_assembly"):
                        output[:, w.start:w.end] = fetched
                audit.records[key] = dict(layer=layer, tensor_type=name, full_sequence_length=sequence_length,
                    native_projection_rows=len(index), reconstructed_projection_rows=0,
                    reused_projection_rows=reused, expected_saved_projection_positions=reused,
                    native_call_count=int(fresh is not None), native_positions=fresh_positions.tolist(),
                    component_scope="total_qkv")
                audit.projections.setdefault(layer, {})[name] = output.detach()
                return output
        return forward

    try:
        for layer in range(len(adapter.layers)):
            for name, module in adapter.projection_modules(layer).items():
                validate_projection(module, adapter_name)
                if module._forward_hooks or module._forward_pre_hooks or "forward" in module.__dict__:
                    raise ValueError("Mixed path requires uninstrumented projections")
                if any(layer not in hit.entry.tensors for hit in hits):
                    raise ValueError("Cached TOTAL QKV must cover all layers")
                originals.append(module)
                module.forward = replacement(layer, name, module, module.forward)
        yield audit
        audit_projection_records(list(audit.records.values()), sequence_length, len(adapter.layers), reused)
    finally:
        for module in originals:
            del module.forward


@contextmanager
def native_events(adapter, book):
    """Passive hooks: observe normal FULL rows; do not perform a second GEMM."""
    handles, spans, records = [], {}, []
    try:
        for layer in range(len(adapter.layers)):
            for name, module in adapter.projection_modules(layer).items():
                def before(module, args, key=(layer, name)):
                    n = args[0].shape[1]
                    records.append(dict(layer=key[0], tensor_type=key[1], native_projection_rows=n,
                        reused_projection_rows=0, native_call_count=1, native_positions=list(range(n))))
                    spans[key] = book.span("native_qkv")
                    spans[key].__enter__()
                def after(module, args, output, key=(layer, name)):
                    spans.pop(key).__exit__(None, None, None)
                handles.append(module.register_forward_pre_hook(before))
                handles.append(module.register_forward_hook(after))
        yield records
    finally:
        for handle in handles:
            handle.remove()


def native_request(model, tokenizer, adapter, text, user, detailed):
    import torch
    from semcache.utils.timing import CPUWallTimer, CUDATimer
    device = next(model.parameters()).device
    book = Intervals(device)
    synchronize(device)
    start = perf_counter_ns()
    model.set_adapter(user)
    model.eval()
    with CPUWallTimer() as token_timer:
        ids = list(tokenizer(text)["input_ids"])
    inputs = dict(input_ids=torch.tensor([ids], device=device), use_cache=False, output_attentions=True)
    gpu = CUDATimer(device) if detailed and str(device).startswith("cuda") else None
    with CPUWallTimer() as prefill:
        with native_events(adapter, book) if detailed else nullcontext([]) as records:
            with torch.inference_mode(), gpu if gpu else nullcontext():
                output = model(**inputs)
        synchronize(device)
    request_ms = (perf_counter_ns() - start) / 1e6
    timing = dict(request_wall_ms=request_ms, prefill_wall_ms=prefill.elapsed_ms,
                  prefill_gpu_ms=gpu.resolve(synchronize=False) if gpu else None,
                  tokenization_ms=token_timer.elapsed_ms)
    if detailed:
        timing.update(book.export())
    # Logit transfer and correctness are outside the closed request.
    logits = output.logits.detach().cpu()
    return dict(logits=logits, ids=ids, timing=timing, projection_records=records,
                cache_fetch_calls=0, h2d_cache_bytes=0)


def engine_request(engine, text, user, query_id, execution_mode, detailed):
    import semcache.semcache_engine as module
    from semcache.edgelora.mixed_projection import mixed_projection_path
    from semcache.utils.timing import close_request_timing
    device = next(engine.model.parameters()).device
    book, captured = Intervals(device), {}

    def close(timer, registry, collect):
        synchronize(device)
        values, scopes = close_request_timing(timer, registry, collect)
        captured["request_wall_ms"] = timer.elapsed_ms
        return values, scopes

    def mixed(*args, **kwargs):
        return detailed_mixed(*args, **kwargs, book=book) if detailed else mixed_projection_path(*args, **kwargs)

    synchronize(device)
    with patch.object(module, "mixed_projection_path", mixed), patch.object(module, "close_request_timing", close):
        result = engine.query(text, user, query_id, execution_mode=execution_mode, collect_timing=detailed)
    timing = result["summary"]["timing"]
    timing.update(captured)
    if detailed:
        timing.update(book.export())
        # The engine's legacy mixed timer is empty for our detailed executor.
        timing["mixed_qkv_execution_ms"] = timing["mixed_executor_gpu_ms"]
        timing["qkv_execution_ms"] = timing["mixed_executor_gpu_ms"]
    else:
        # Unsynchronized engine prefill wall is not a valid GPU elapsed measure.
        timing.update(prefill_wall_ms=None, prefill_gpu_ms=None)
    return dict(**result, cache_fetch_calls=book.cache_fetch_calls if detailed else None,
                h2d_cache_bytes=book.h2d_cache_bytes if detailed else None)


def memory_snapshot(device):
    import torch
    if not str(device).startswith("cuda"):
        return dict(cuda_available=False)
    free, total = torch.cuda.mem_get_info(device)
    return dict(free_bytes=free, total_bytes=total, allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def raw_storage_audit(cache, storage):
    """Inspect resident ownership outside timing; no encoded payload is accepted."""
    import torch
    tensors, devices, dtypes = 0, set(), set()
    expected_device = torch.device(storage)
    for entry in cache.entries.values():
        if not entry.tensors or entry.qkv_metadata.get("component_scope") != "total_qkv":
            raise AssertionError("R1 requires resident raw TOTAL QKV")
        for qkv in entry.tensors.values():
            if len(qkv) != 3:
                raise AssertionError("Expected three raw tensors")
            for tensor in qkv:
                if (not isinstance(tensor, torch.Tensor) or tensor.is_quantized
                        or not tensor.is_floating_point() or tensor.device != expected_device):
                    raise AssertionError("Raw storage dtype/device contract violated")
                tensors += 1
                devices.add(str(tensor.device))
                dtypes.add(str(tensor.dtype))
    return dict(raw_tensor_count=tensors, devices=sorted(devices), dtypes=sorted(dtypes),
                resident_bytes=cache.physical_tensor_bytes, codec_calls=0)


def memory_preflight(model, tokens, capacity, storage, device):
    """Conservative screening, not an allocator/OOM guarantee."""
    result = memory_snapshot(device)
    config = model.config
    # Full fp32 attention plus double precision reducer workspace, QKV captures,
    # activations and logits; include temporary admission overflow + fragmentation.
    attention = config.num_hidden_layers * config.num_attention_heads * tokens * tokens * 12
    activations = config.num_hidden_layers * tokens * config.hidden_size * 32
    logits = tokens * config.vocab_size * 8
    cache = 2 * capacity if str(storage).startswith("cuda") else 0
    required = attention + activations + logits + cache + 512 * 1024**2
    result.update(estimated_additional_bytes=required, cache_capacity_bytes=capacity,
                  estimator="screening only; 12-byte attention, 32-byte activations, 8-byte logits, 2x GPU cache, 512MiB reserve")
    if str(device).startswith("cuda"):
        # Empty allocator blocks are reusable; free VRAM alone understates headroom.
        available = result["free_bytes"] + result["reserved_bytes"] - result["allocated_bytes"]
        result["available_with_allocator_bytes"] = available
        if required > available:
            raise MemoryError(f"R1 memory preflight rejected configuration: {result}")
    return result
