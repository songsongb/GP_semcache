"""M8 schemas, aggregation, model guards, and analytical communication."""
from __future__ import annotations

import math
import platform
import socket
import hashlib
from datetime import datetime, timezone
from statistics import mean, median, pstdev

MODES = ("NATIVE_NO_CACHE", "SEMCACHE_LOOKUP_NO_REUSE", "SEMCACHE_PHYSICAL_REUSE")
MODEL_DEFAULTS = {
    "facebook/opt-125m": dict(dtype="float32", warmup_runs=5, measured_runs=20, large=False),
    "facebook/opt-2.7b": dict(dtype="float16", warmup_runs=3, measured_runs=10, large=False),
    "facebook/opt-6.7b": dict(dtype="float16", warmup_runs=2, measured_runs=5, large=True),
}

TIMING_FIELDS = (
    "semantic_encode_ms", "cluster_assign_update_ms", "tokenization_ms",
    "subsequence_extract_ms", "cache_lookup_ms", "hit_selection_ms",
    "cache_policy_ms", "chu_update_ms", "pbr_update_ms",
    "cache_materialization_ms", "base_qkv_projection_ms",
    "lora_qkv_projection_ms", "qkv_execution_ms", "mixed_qkv_execution_ms", "attention_ms",
    "remaining_transformer_ms", "prefill_gpu_ms", "request_wall_ms",
)

RAW_FIELDS = (
    "experiment_id", "timestamp", "hostname", "gpu_name", "model_id",
    "model_revision", "tokenizer_revision", "dtype", "mode", "query_id",
    "user_id", "adapter_name", "prompt_tokens", "reused_tokens",
    "recomputed_tokens", "token_reuse_ratio", "block_hits", "candidate_blocks",
    "physical_reuse_used", "projection_skip_used", "warmup_runs",
    "measured_runs", "repeat_index", *TIMING_FIELDS,
    "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "model_weight_memory_bytes",
    "max_abs_logit_diff", "relative_l2_logit_diff", "last_position_kl",
    "argmax_agreement", "saved_communication_elements",
    "saved_communication_bytes", "estimated_communication_ms",
    "measured_or_analytical", "communication_metric_source",
    "safe_reuse_claimed", "timing_scopes",
    "controlled_exact_parity", "repetition_state_semantics",
    "warmup_state_semantics", "trace_position", "seed",
    "attention_implementation", "prompt_token_ids_sha256",
    "cuda_allocated_before_bytes", "cuda_reserved_before_bytes",
    "incremental_peak_allocated_bytes", "incremental_peak_reserved_bytes",
)

TRAINING_FIELDS = (
    "experiment_id", "timestamp", "hostname", "gpu_name", "model_id",
    "model_revision", "tokenizer_revision", "dtype", "rank", "target_modules",
    "sample_count", "epochs", "sequence_length", "batch_size",
    "gradient_checkpointing", "mixed_precision", "total_training_wall_s",
    "epoch_time_s", "step_time_ms", "samples_per_second", "tokens_per_second",
    "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
    "trainable_parameter_count", "adapter_checkpoint_bytes",
    "measured_or_estimated", "reproduction_choices",
)


def model_config(model_id, *, allow_large_model=False, **overrides):
    if model_id not in MODEL_DEFAULTS:
        raise ValueError(f"Unsupported M8 model: {model_id}")
    result = dict(model_id=model_id, **MODEL_DEFAULTS[model_id])
    result.update({k: v for k, v in overrides.items() if v is not None})
    if result["large"] and not allow_large_model:
        raise PermissionError("OPT-6.7B requires --allow-large-model")
    if result["large"] and (result["warmup_runs"] > 2 or result["measured_runs"] > 5):
        raise ValueError("OPT-6.7B scale check is capped at 2 warmups and 5 measured runs")
    return result


def repetition_schedule(warmup_runs, measured_runs):
    """Each item denotes a fresh, independently reconstructed trace state."""
    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("warmup_runs must be nonnegative and measured_runs positive")
    return [dict(phase="warmup", repeat_index=i) for i in range(warmup_runs)] + [
        dict(phase="measured", repeat_index=i) for i in range(measured_runs)]


def token_ids_sha256(token_ids):
    payload = ",".join(str(int(token)) for token in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def incremental_memory(allocated_before, reserved_before, peak_allocated, peak_reserved):
    if allocated_before is None:
        return dict(incremental_peak_allocated_bytes=None, incremental_peak_reserved_bytes=None)
    return dict(incremental_peak_allocated_bytes=max(0, peak_allocated - allocated_before),
                incremental_peak_reserved_bytes=max(0, peak_reserved - reserved_before))


def mode_identity(model_revision, tokenizer_revision, dtype, token_hash, adapter_name,
                  seed, attention_implementation):
    """Fields that must be equal across modes; mode itself is intentionally absent."""
    return dict(model_revision=model_revision, tokenizer_revision=tokenizer_revision,
        dtype=dtype, prompt_token_ids_sha256=token_hash, adapter_name=adapter_name,
        seed=seed, attention_implementation=attention_implementation)


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def aggregate_raw(rows):
    """One row per experiment/mode/query/metric with distribution statistics."""
    groups = {}
    for row in rows:
        key = (row["experiment_id"], row["model_id"], row["mode"], row["query_id"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, items in sorted(groups.items()):
        for field in TIMING_FIELDS:
            values = [r.get(field) for r in items if r.get(field) is not None]
            output.append(dict(experiment_id=key[0], model_id=key[1], mode=key[2],
                query_id=key[3], metric=field, count=len(values), mean=mean(values) if values else None,
                p50=median(values) if values else None, p95=percentile(values, .95),
                std=pstdev(values) if values else None, min=min(values) if values else None,
                max=max(values) if values else None, unit="ms", metric_source="MEASURED"))
    return output


def raw_record(**values):
    row = {field: None for field in RAW_FIELDS}
    row.update(values)
    unknown = set(row) - set(RAW_FIELDS)
    if unknown:
        raise ValueError(f"Unknown M8 raw fields: {sorted(unknown)}")
    row["timestamp"] = row["timestamp"] or datetime.now(timezone.utc).isoformat()
    row["hostname"] = row["hostname"] or socket.gethostname()
    row["safe_reuse_claimed"] = False
    return row


def training_record(**values):
    row = {field: None for field in TRAINING_FIELDS}
    row.update(values)
    unknown = set(row) - set(TRAINING_FIELDS)
    if unknown:
        raise ValueError(f"Unknown M8 training fields: {sorted(unknown)}")
    row["timestamp"] = row["timestamp"] or datetime.now(timezone.utc).isoformat()
    row["hostname"] = row["hostname"] or socket.gethostname()
    return row


def analytical_communication(elements, element_bytes, bandwidth_bytes_per_s=None):
    saved_bytes = int(elements) * int(element_bytes)
    return dict(saved_communication_elements=int(elements), saved_communication_bytes=saved_bytes,
        estimated_communication_ms=(saved_bytes / bandwidth_bytes_per_s * 1000
                                    if bandwidth_bytes_per_s else None),
        measured_or_analytical="ANALYTICAL / SIMULATED")


def environment_record(torch_module=None):
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
                  platform=platform.platform(), python=platform.python_version())
    if torch_module is not None:
        result.update(torch_version=torch_module.__version__, cuda_available=torch_module.cuda.is_available(),
            cuda_version=torch_module.version.cuda,
            gpu_name=torch_module.cuda.get_device_name() if torch_module.cuda.is_available() else None)
    return result
