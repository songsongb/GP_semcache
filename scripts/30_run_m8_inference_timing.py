#!/usr/bin/env python3
"""Run the deterministic M8 three-mode inference microbenchmark."""
import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semcache.cache.global_cache import GlobalCache
from semcache.metrics.inference import (begin_cuda_memory_measurement,
    finish_cuda_memory_measurement, model_weight_bytes, native_request, record_from_engine)
from semcache.metrics.m8 import (MODES, aggregate_raw, environment_record, model_config,
    cache_transfer_path, mode_identity, raw_record, repetition_schedule, token_ids_sha256)
from semcache.models.loader import load_model
from semcache.models.lora_fixtures import create_controlled_users
from semcache.models.model_adapter import OPTModelAdapter
from semcache.semantic.encoder import TinyBERTSemanticEncoder
from semcache.semantic.intent_clusterer import IntentClusterer
from semcache.semcache_engine import SemCacheEngine
from semcache.utils.io import write_csv, write_json, write_jsonl
from semcache.utils.seed import seed_everything

WORKLOAD = (
    ("cold_miss", "user_a", "Please find a hotel in Cambridge near the station."),
    ("same_user_exact", "user_a", "Please find a hotel in Cambridge near the station."),
    ("cross_user_exact", "user_b", "Please find a hotel in Cambridge near the station."),
    ("unrelated", "user_a", "Book an Italian restaurant in central London tonight."),
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/opt-125m", choices=list(__import__("semcache.metrics.m8", fromlist=["MODEL_DEFAULTS"]).MODEL_DEFAULTS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype")
    p.add_argument("--warmup-runs", type=int)
    p.add_argument("--measured-runs", type=int)
    p.add_argument("--allow-large-model", action="store_true")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--revision", help="Optional requested HF revision; resolved commits are always recorded")
    p.add_argument("--tokenizer-revision")
    p.add_argument("--bandwidth-gbps", type=float)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/m8")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = model_config(args.model, allow_large_model=args.allow_large_model, dtype=args.dtype,
                       warmup_runs=args.warmup_runs, measured_runs=args.measured_runs)
    workload = WORKLOAD[:3] if cfg["large"] else WORKLOAD
    print(f"M8 work summary: model={args.model}, conditions={len(workload)}, modes=3, "
          f"warmups={cfg['warmup_runs']}, measured={cfg['measured_runs']}, dtype={cfg['dtype']}")
    seed_everything(42)
    model, tokenizer, metadata = load_model(dict(name=args.model, tokenizer=args.model,
        revision=args.revision, tokenizer_revision=args.tokenizer_revision or args.revision,
        dtype=cfg["dtype"], device=args.device, local_files_only=not args.allow_download,
        attention_implementation="eager"))
    model, lora_metadata = create_controlled_users(model)
    adapter = OPTModelAdapter(model)
    execution_device = str(next(model.parameters()).device)
    configured_cache_storage_device = "cpu"
    encoder = TinyBERTSemanticEncoder(device=args.device, dtype=cfg["dtype"],
                                      local_files_only=not args.allow_download)
    anchors = [WORKLOAD[0][2], WORKLOAD[-1][2]]
    anchor_vectors = encoder.encode(anchors)

    def engine():
        clusterer = IntentClusterer(2, initialization="first_k", update_mode="immediate_eq9")
        clusterer.initialize(anchor_vectors)
        return SemCacheEngine(model, tokenizer, adapter, encoder, clusterer,
            GlobalCache(256 * 1024 * 1024), metadata=metadata,
            storage_device=configured_cache_storage_device)

    import torch
    # Output provenance is read back from the runtime engine rather than copied
    # from a reporting constant.
    cache_storage_device = str(engine().storage_device)
    transfer_path = cache_transfer_path(cache_storage_device, execution_device)
    env = environment_record(torch)
    env.update(model=metadata, semantic_encoder=encoder.metadata, lora=lora_metadata,
               benchmark_kind="microbenchmark", safe_reuse_claimed=False,
               physical_cache_storage_device=cache_storage_device,
               cache_transfer_path=transfer_path,
               warmup_runs=cfg["warmup_runs"], measured_runs=cfg["measured_runs"])
    experiment_id = f"m8-{uuid.uuid4().hex[:12]}"
    gpu_name = env.get("gpu_name")
    weights = model_weight_bytes(model)
    bandwidth = args.bandwidth_gbps * 1e9 / 8 if args.bandwidth_gbps else None
    rows = []
    expected_identity = {}
    for mode in MODES:
        for scheduled in repetition_schedule(cfg["warmup_runs"], cfg["measured_runs"]):
            repeat, phase = scheduled["repeat_index"], scheduled["phase"]
            # A new engine reconstructs identical centroids, empty cache, and
            # cache-policy counters for every trace. Previous warmups are discarded.
            semcache = engine() if mode != "NATIVE_NO_CACHE" else None
            for trace_position, (condition, user, text) in enumerate(workload):
                reference = native_request(model, tokenizer, text, user, collect_quality=False)
                reference_hash = token_ids_sha256(reference["ids"])
                allocated_before, reserved_before = begin_cuda_memory_measurement(torch, args.device)
                if mode == "NATIVE_NO_CACHE":
                    measured = native_request(model, tokenizer, text, user)
                    if token_ids_sha256(measured["ids"]) != reference_hash:
                        raise AssertionError("Native/reference token IDs differ")
                    q = measured["quality"]
                    row = raw_record(experiment_id=experiment_id, gpu_name=gpu_name,
                        model_id=args.model, model_revision=metadata.get("resolved_model_revision"),
                        tokenizer_revision=metadata.get("resolved_tokenizer_revision"), dtype=metadata["dtype"],
                        mode=mode, query_id=condition, user_id=user, adapter_name=user,
                        prompt_tokens=len(measured["ids"]), reused_tokens=0,
                        recomputed_tokens=len(measured["ids"]), token_reuse_ratio=0.0,
                        block_hits=0, candidate_blocks=0, physical_reuse_used=False,
                        projection_skip_used=False, warmup_runs=cfg["warmup_runs"],
                        measured_runs=cfg["measured_runs"], repeat_index=repeat,
                        tokenization_ms=measured["tokenization_ms"],
                        prefill_wall_ms=measured["prefill_wall_ms"],
                        prefill_gpu_ms=measured["prefill_gpu_ms"],
                        prefill_host_overhead_ms=measured["prefill_host_overhead_ms"],
                        request_wall_ms=measured["request_wall_ms"], model_weight_memory_bytes=weights,
                        correctness_reference_ms=reference["request_wall_ms"],
                        quality_diagnostics_ms=measured["quality_diagnostics_ms"],
                        max_abs_logit_diff=q["max_abs_logit_diff"], relative_l2_logit_diff=q["relative_l2_logit_diff"],
                        last_position_kl=q["last_position_kl_baseline_to_injected"],
                        argmax_agreement=q["last_argmax_agreement"], measured_or_analytical="MEASURED",
                        communication_metric_source="ANALYTICAL / SIMULATED",
                        physical_cache_storage_device=cache_storage_device,
                        cache_transfer_path=transfer_path,
                        controlled_exact_parity=True,
                        timing_scopes={"request_wall_ms": {"timing_scope": "complete native request",
                            "timing_parent": None, "inclusive_or_exclusive": "inclusive", "clock": "cpu_perf_counter_ns"},
                            "prefill_wall_ms": {"timing_scope": "model forward plus CUDA timing resolution",
                            "timing_parent": "request_wall_ms", "inclusive_or_exclusive": "exclusive_top_level",
                            "clock": "cpu_perf_counter_ns"},
                            "prefill_gpu_ms": {"timing_scope": "one OPT prefill forward", "timing_parent": "prefill_wall_ms",
                            "inclusive_or_exclusive": "inclusive", "clock": "cuda_event"},
                            "prefill_host_overhead_ms": {"timing_scope": "prefill wall minus GPU diagnostic approximation",
                            "timing_parent": "prefill_wall_ms", "inclusive_or_exclusive": "derived_do_not_sum",
                            "clock": "derived_cpu_wall_minus_cuda_event"}})
                else:
                    result = semcache.query(text, user, condition, execution_mode=mode,
                        collect_timing=True, baseline_logits=reference["logits"])
                    if token_ids_sha256(result["summary"]["token_ids"]) != reference_hash:
                        raise AssertionError("Native/SemCache token IDs differ")
                    row = record_from_engine(result, experiment_id=experiment_id, repeat_index=repeat,
                        warmup_runs=cfg["warmup_runs"], measured_runs=cfg["measured_runs"],
                        model_metadata=metadata, gpu_name=gpu_name, condition=condition,
                        weight_bytes=weights,
                        bandwidth_bytes_per_s=bandwidth)
                row.update(finish_cuda_memory_measurement(torch, args.device,
                    allocated_before, reserved_before))
                row["correctness_reference_ms"] = reference["request_wall_ms"]
                row.update(repetition_state_semantics=("stateless" if mode == "NATIVE_NO_CACHE"
                           else "reconstructed_precondition"),
                    warmup_state_semantics="fresh_discarded_trace",
                    trace_position=trace_position, seed=42, attention_implementation="eager",
                    prompt_token_ids_sha256=reference_hash)
                identity = mode_identity(row["model_revision"], row["tokenizer_revision"],
                    row["dtype"], row["prompt_token_ids_sha256"], row["adapter_name"],
                    row["seed"], row["attention_implementation"])
                if condition in expected_identity and identity != expected_identity[condition]:
                    raise AssertionError(f"Mode execution identity differs for {condition}")
                expected_identity.setdefault(condition, identity)
                if phase == "measured":
                    rows.append(row)
    write_jsonl(args.output_dir / "inference_raw.jsonl", rows)
    write_csv(args.output_dir / "inference_summary.csv", aggregate_raw(rows))
    write_json(args.output_dir / "inference_environment.json", env)
    print(f"Saved {len(rows)} raw records to {args.output_dir}; safe_reuse_claimed=false")


if __name__ == "__main__":
    main()
