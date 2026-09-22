"""Reusable M8 inference measurement helpers."""
from __future__ import annotations

from semcache.metrics.alignment import artifact_alignment
from semcache.evaluation.logit_metrics import compare_logits
from semcache.metrics.m8 import (cache_transfer_path, incremental_memory,
                                 raw_record, token_ids_sha256)
from semcache.metrics.timing import CPUWallTimer, CUDATimer


def model_weight_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def begin_cuda_memory_measurement(torch, device):
    if not (torch.cuda.is_available() and str(device).startswith("cuda")):
        return None, None
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    return allocated, reserved


def finish_cuda_memory_measurement(torch, device, allocated_before, reserved_before):
    if allocated_before is None:
        return dict(cuda_allocated_before_bytes=None, cuda_reserved_before_bytes=None,
            peak_cuda_allocated_bytes=None, peak_cuda_reserved_bytes=None,
            **incremental_memory(None, None, None, None))
    # The enclosing CUDA timing event has already synchronized completion.
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return dict(cuda_allocated_before_bytes=allocated_before,
        cuda_reserved_before_bytes=reserved_before,
        peak_cuda_allocated_bytes=peak_allocated, peak_cuda_reserved_bytes=peak_reserved,
        **incremental_memory(allocated_before, reserved_before, peak_allocated, peak_reserved))


def native_request(model, tokenizer, text, user_id, *, collect_quality=True):
    """Normal PEFT/OPT prefill: no SemCache frontend, lookup, or patched projection."""
    import torch
    if hasattr(model, "set_adapter"):
        model.set_adapter(user_id)
    model.eval()
    wall = CPUWallTimer()
    with wall:
        token_timer = CPUWallTimer()
        with token_timer:
            ids = list(tokenizer(text)["input_ids"])
        inputs = dict(input_ids=torch.tensor([ids], device=next(model.parameters()).device),
                      use_cache=False, output_attentions=True)
        cuda_timer = CUDATimer(inputs["input_ids"].device) if inputs["input_ids"].is_cuda else None
        prefill_wall = CPUWallTimer()
        prefill_wall.__enter__()
        with torch.inference_mode():
            if cuda_timer:
                with cuda_timer:
                    output = model(**inputs)
            else:
                output = model(**inputs)
        if cuda_timer:
            _ = cuda_timer.elapsed_ms
        prefill_wall.__exit__(None, None, None)
    quality_timer = CPUWallTimer()
    with quality_timer:
        logits = output.logits.detach().cpu()
        quality = compare_logits(logits, logits, 0) if collect_quality else {}
    return dict(ids=ids, logits=logits, tokenization_ms=token_timer.elapsed_ms,
                prefill_wall_ms=prefill_wall.elapsed_ms,
                prefill_gpu_ms=cuda_timer.elapsed_ms if cuda_timer else None,
                prefill_host_overhead_ms=(prefill_wall.elapsed_ms - cuda_timer.elapsed_ms
                                          if cuda_timer else None),
                request_wall_ms=wall.elapsed_ms, quality_diagnostics_ms=quality_timer.elapsed_ms,
                quality=quality)


def record_from_engine(result, *, experiment_id, repeat_index, warmup_runs,
                       measured_runs, model_metadata, gpu_name, condition,
                       peak_allocated=None, peak_reserved=None, weight_bytes=None,
                       bandwidth_bytes_per_s=None):
    s, timing = result["summary"], result["summary"].get("timing", {})
    elements = s["paper_estimated_comm_elements_saved"]
    saved_bytes = s["paper_estimated_comm_bytes_saved"]
    comm = dict(saved_communication_elements=elements, saved_communication_bytes=saved_bytes,
        estimated_communication_ms=(saved_bytes / bandwidth_bytes_per_s * 1000
                                    if bandwidth_bytes_per_s else None),
        communication_metric_source="ANALYTICAL / SIMULATED")
    return raw_record(**artifact_alignment(s), experiment_id=experiment_id, gpu_name=gpu_name,
        model_id=model_metadata["model"], model_revision=model_metadata.get("resolved_model_revision"),
        tokenizer_revision=model_metadata.get("resolved_tokenizer_revision"), dtype=s["dtype"],
        mode=s["mode"], query_id=s["query_id"], user_id=s["user_id"], adapter_name=s["adapter_name"],
        prompt_tokens=s["query_token_count"], reused_tokens=s["reused_unique_token_count"],
        recomputed_tokens=s["recomputed_tokens"], token_reuse_ratio=s["token_reuse_ratio"],
        block_hits=s["block_hit_count"], candidate_blocks=s["candidate_windows"],
        physical_reuse_used=s["physical_reuse_used"], projection_skip_used=s["projection_skip_used"],
        attention_impact_block_count=s["attention_impact_block_count"],
        executed_nonoverlap_hits=s.get("executed_nonoverlap_hits"),
        chu_update_count=s.get("chu_update_count"),
        pbr_updates_this_query=s.get("pbr_updates_this_query"),
        warmup_runs=warmup_runs, measured_runs=measured_runs, repeat_index=repeat_index,
        peak_cuda_allocated_bytes=peak_allocated, peak_cuda_reserved_bytes=peak_reserved,
        model_weight_memory_bytes=weight_bytes, max_abs_logit_diff=s["max_abs_logit_diff"],
        relative_l2_logit_diff=s["relative_l2_logit_diff"],
        last_position_kl=s["last_position_kl_baseline_to_injected"],
        argmax_agreement=s["last_argmax_agreement"], timing_scopes=s.get("timing_scopes"),
        controlled_exact_fixture=condition == "same_user_exact",
        correctness_validation_status=("correctness_not_validated_for_lengths_ge_64"
            if s["mode"] == "SEMCACHE_PHYSICAL_REUSE" else "diagnostic_recorded"),
        reuse_block_provenance=s.get("reuse_block_provenance"),
        admission_audit=s.get("admission_audit"),
        deduplication_audit=s.get("deduplication_audit"),
        admission_candidate_count=s.get("admission_candidate_count"),
        admitted_block_count=s.get("admitted_block_count"),
        rejected_block_count=s.get("rejected_block_count"),
        deduplicated_window_count=s.get("deduplicated_window_count"),
        prompt_token_ids_sha256=token_ids_sha256(s.get("token_ids", [])),
        physical_cache_storage_device=str(s["physical_storage_device"]),
        cache_transfer_path=cache_transfer_path(s["physical_storage_device"], model_metadata["device"]),
        measured_or_analytical="MEASURED",
        **{name: timing.get(name) for name in raw_record().keys() if name.endswith("_ms") and name not in comm},
        **comm)
