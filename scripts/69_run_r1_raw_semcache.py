#!/usr/bin/env python3
"""Independent raw-only OPT prefill benchmark. No downloads unless requested."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semcache.metrics.r1 import (MODES, ENGINE_MODES, atomic_json, digest,
    audit_projection_records, raw_only_guard, write_reports, timing_scopes)


def parse_args(argv=None, extension=None):
    from semcache.metrics.alignment import add_alignment_arguments, validate_alignment_arguments
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("facebook/opt-125m", "facebook/opt-2.7b"), default="facebook/opt-125m")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), default="float16")
    p.add_argument("--revision")
    p.add_argument("--tokenizer-revision")
    p.add_argument("--encoder-revision", default="main")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--prompt-lengths", default="32,64,128,256")
    p.add_argument("--cache-devices", default="cpu", help="cpu, cuda, or cpu,cuda; cuda uses --device")
    p.add_argument("--cache-capacity-mib", type=int, default=256)
    p.add_argument("--instrumentation", choices=("both", "minimal", "breakdown"), default="both")
    p.add_argument("--warmup-runs", type=int, default=3)
    p.add_argument("--measured-runs", type=int, default=10)
    p.add_argument("--smoke", action="store_true", help="Permit fewer repetitions; never report as a full benchmark")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--output-dir", type=Path, required=True, help="New isolated directory under results/r1 recommended")
    p.add_argument("--resume", action="store_true", help="Resume atomic completed pairing groups; rewarm after restart")
    add_alignment_arguments(p)
    if extension:
        extension.add_arguments(p)
    args = p.parse_args(argv)
    validate_alignment_arguments(args)
    try:
        args.lengths = [int(x) for x in args.prompt_lengths.split(",")]
    except ValueError:
        p.error("--prompt-lengths must contain integers")
    if not args.lengths or len(set(args.lengths)) != len(args.lengths) or any(x not in (32,64,128,256) for x in args.lengths):
        p.error("R1-A permits unique lengths 32,64,128,256 only; 512+ requires a later preflight study")
    args.storages = args.cache_devices.split(",")
    if not args.storages or len(set(args.storages)) != len(args.storages) or any(x not in ("cpu", "cuda") for x in args.storages):
        p.error("--cache-devices must be cpu, cuda, or cpu,cuda")
    if "cuda" in args.storages and not args.device.startswith("cuda"):
        p.error("GPU cache requires CUDA execution")
    if args.warmup_runs < 0 or args.measured_runs < 1 or args.cache_capacity_mib <= 0:
        p.error("Invalid repetition count or cache capacity")
    if not args.smoke and (args.warmup_runs < 3 or args.measured_runs < 10):
        p.error("Use >=3 warmups and >=10 measured repeats, or explicitly --smoke")
    if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        p.error("Parity tolerances must be nonnegative")
    if extension:
        extension.validate(args, p)
    return args


def command_output(command):
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return "unavailable: " + str(exc)


def code_provenance():
    paths = sorted((ROOT / "src/semcache").rglob("*.py")) + [Path(__file__).resolve()]
    return dict(git_head=command_output(["git", "rev-parse", "HEAD"]),
        git_branch=command_output(["git", "branch", "--show-current"]),
        git_status=command_output(["git", "status", "--short"]),
        files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def source_signature(engine):
    return digest([dict(key=e.key, positions=e.positions, size=e.size_bytes, metadata=e.qkv_metadata,
                        tensor_layout=[(i, str(t.dtype), list(t.shape)) for i,qkv in sorted(e.tensors.items()) for t in qkv])
                   for _, e in sorted(engine.cache.entries.items())])


def run(args, manifest, guard, extension=None):
    import torch
    from semcache.models.loader import load_model
    from semcache.models.lora_fixtures import create_controlled_users
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.semantic.encoder import TinyBERTSemanticEncoder
    from semcache.semantic.intent_clusterer import IntentClusterer
    from semcache.cache.global_cache import GlobalCache
    from semcache.semcache_engine import SemCacheEngine
    from semcache.metrics.scaling_workload import controlled_length_trace, subsequence_occurrence_report
    from semcache.metrics.r1_runtime import (engine_request, native_request, synchronize,
                                            memory_preflight, memory_snapshot, raw_storage_audit)
    from semcache.evaluation.logit_metrics import compare_logits
    from semcache.utils.seed import seed_everything

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU benchmark NOT EXECUTED")
    seed_everything(args.seed)
    start = perf_counter_ns()
    model, tokenizer, metadata = load_model(dict(name=args.model, tokenizer=args.model,
        revision=args.revision, tokenizer_revision=args.tokenizer_revision or args.revision,
        dtype=args.dtype, device=args.device, local_files_only=not args.allow_download,
        attention_implementation="eager"))
    if getattr(model, "is_quantized", False) or getattr(model.config, "quantization_config", None):
        raise ValueError("R1 requires an unquantized model")
    model, lora = create_controlled_users(model)
    adapter = OPTModelAdapter(model)
    synchronize(args.device)
    initialization_ms = (perf_counter_ns() - start) / 1e6
    start = perf_counter_ns()
    encoder = TinyBERTSemanticEncoder(device=args.device, dtype=args.dtype,
        revision=args.encoder_revision, local_files_only=not args.allow_download)
    synchronize(args.device)
    encoder_initialization_ms = (perf_counter_ns()-start)/1e6
    runtime = dict(model=metadata, lora=lora, encoder=encoder.metadata,
        torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        driver=command_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"]),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if manifest.get("runtime") and manifest["runtime"] != runtime:
        raise ValueError("Resume runtime/model revision identity differs")
    manifest.update(runtime=runtime, model_and_adapter_initialization_ms=initialization_ms,
                    semantic_encoder_initialization_ms=encoder_initialization_ms)
    atomic_json(args.output_dir / "manifest.json", manifest)
    rows = []
    for checkpoint in sorted((args.output_dir / "checkpoints").glob("*.json")):
        bundle = json.loads(checkpoint.read_text())
        if bundle["identity"] != manifest["identity"]:
            raise ValueError("Checkpoint identity differs")
        rows.extend(bundle["rows"])
    capacity = args.cache_capacity_mib * 1024**2
    passes = ["minimal", "breakdown"] if args.instrumentation == "both" else [args.instrumentation]
    schedule = [("warmup", i) for i in range(args.warmup_runs)] + [("measured", i) for i in range(args.measured_runs)]
    for length in args.lengths:
        trace = controlled_length_trace(tokenizer, length)
        anchors = encoder.encode([trace[0]["text"], trace[-1]["text"]])
        def engine(storage):
            clusterer = IntentClusterer(2, initialization="first_k", update_mode=args.cluster_update_mode,
                update_interval=args.cluster_update_interval)
            clusterer.initialize(anchors)
            return SemCacheEngine(model, tokenizer, adapter, encoder, clusterer, GlobalCache(capacity),
                storage_device=storage, metadata=metadata, seed=args.seed, impact_reducer=args.impact_reducer,
                rho=args.rho, history_lambda=args.history_lambda,
                pbr_interval_queries=args.pbr_interval if args.pbr_mode == "interval" else None,
                **(extension.engine_options(variant) if extension else {}))
        for phase, repeat in schedule:
            # Alternate pass/storage order across repetitions; rotate mode order.
            variants = extension.variants(args) if extension else [None]
            conditions = [(s, p, v) for s in args.storages for p in passes for v in variants]
            if repeat % 2:
                conditions.reverse()
            for storage_name, instrumentation, variant in conditions:
                name = f"{length}-{phase}-{repeat}-{storage_name}-{instrumentation}"
                if extension:
                    name += f"-{variant}"
                checkpoint = args.output_dir / "checkpoints" / (name + ".json")
                if phase == "measured" and checkpoint.exists():
                    print("RESUME completed", name, flush=True)
                    continue
                storage = str(next(model.parameters()).device) if storage_name == "cuda" else "cpu"
                detailed = instrumentation == "breakdown"
                preflight = memory_preflight(model, max(x["actual_prompt_tokens"] for x in trace), capacity, storage, args.device)
                manifest["active_group"] = name
                manifest["last_preflight"] = preflight
                atomic_json(args.output_dir / "manifest.json", manifest)
                batch = []
                for item in trace:
                    condition, text, user = item["condition"], item["text"], item["user_id"]
                    # Same untimed native reference for every mode of this pair.
                    reference = native_request(model, tokenizer, adapter, text, user, False)
                    expected_ids = item["token_ids"]
                    if reference["ids"] != expected_ids:
                        raise AssertionError("Constructed/reference token IDs differ")
                    mode_order = list(MODES[repeat % len(MODES):] + MODES[:repeat % len(MODES)])
                    for mode in mode_order:
                        semcache, build, signature = None, None, digest([])
                        if mode != "FULL_RECOMPUTE":
                            semcache = engine(storage)
                            if condition != "cold_miss":
                                source = engine_request(semcache, trace[0]["text"], "user_a", "source",
                                    ENGINE_MODES["LOOKUP_NO_REUSE"], True)
                                build = dict(timing=source["summary"]["timing"],
                                    timing_scopes=timing_scopes(source["summary"]["timing"]),
                                    admitted_blocks=source["summary"]["admitted_block_count"],
                                    resident_bytes=semcache.cache.physical_tensor_bytes,
                                    admission_audit=source["summary"]["admission_audit"])
                                if extension:
                                    build['policy_snapshot'] = extension.snapshot(semcache, source)
                                del source
                            signature = source_signature(semcache)
                            raw_storage_audit(semcache.cache, storage)
                        synchronize(args.device)
                        resident_before = semcache.cache.physical_tensor_bytes if semcache else 0
                        if args.device.startswith("cuda"):
                            torch.cuda.reset_peak_memory_stats(args.device)
                        before = memory_snapshot(args.device)
                        if mode == "FULL_RECOMPUTE":
                            result = native_request(model, tokenizer, adapter, text, user, detailed)
                            s = dict(token_ids=result["ids"], reused_unique_token_count=0, block_hit_count=0,
                                     timing=result["timing"], reuse_block_provenance=[], executed_nonoverlap_hits=0)
                            records = result["projection_records"]
                        else:
                            result = engine_request(semcache, text, user, condition, ENGINE_MODES[mode], detailed)
                            s = result["summary"]
                            records = list(result["projection_audit"].values())
                        after = memory_snapshot(args.device)
                        if s["token_ids"] != expected_ids:
                            raise AssertionError("Modes differ in token IDs")
                        reused = s["reused_unique_token_count"]
                        spans = [i for p in s["reuse_block_provenance"] for i in range(p["destination_start"], p["destination_end"])]
                        audit = audit_projection_records(records, len(expected_ids), len(adapter.layers), reused, spans) if records else dict(
                            executed_projection_records=None, skipped_projection_rows_all_layers_roles=0,
                            native_projection_rows_all_layers_roles=None, projection_rows_per_role_per_layer=0)
                        if condition == "cold_miss" and reused:
                            raise AssertionError("Cold request reused projection rows")
                        quality_start = perf_counter_ns()
                        quality = compare_logits(reference["logits"], result["logits"], 0)
                        quality.update(allclose=bool(torch.allclose(reference["logits"], result["logits"], atol=args.atol, rtol=args.rtol)),
                            atol=args.atol, rtol=args.rtol, computed_outside_timing=True,
                            fixture="strict_position_aligned" if mode == "POSITION_ALIGNED_DIAGNOSTIC" else "natural_exact_w3",
                            nonzero_reuse_validated=False)
                        quality["nonzero_reuse_validated"] = bool(reused and quality["allclose"])
                        quality["comparison_ms_outside_request"] = (perf_counter_ns()-quality_start)/1e6
                        provenance = s["reuse_block_provenance"]
                        if mode == "POSITION_ALIGNED_DIAGNOSTIC" and any(not p["positions_identical"] or p["source_adapter"] != user for p in provenance):
                            raise AssertionError("Position diagnostic admitted incompatible source")
                        row = dict(pair_id=f"{name}-{condition}", phase=phase, repeat=repeat,
                            storage="RAW_GPU_DIAGNOSTIC" if storage_name == "cuda" else "RAW_CPU",
                            instrumentation=instrumentation, mode=mode, diagnostic="DIAGNOSTIC" in mode,
                            requested_prompt_tokens=length, actual_prompt_tokens=len(expected_ids), condition=condition,
                            measurement_scope="cold_start_including_admission" if condition == "cold_miss" else "target_after_source_build",
                            adapter=user, token_ids_sha256=digest(expected_ids), input_token_ids=expected_ids,
                            timing=s["timing"], timing_scopes=timing_scopes(s["timing"]),
                            source_build=build, source_signature=signature,
                            reuse_signature=digest(provenance), reuse_provenance=provenance,
                            subsequence_audit=subsequence_occurrence_report(expected_ids),
                            raw_block_hits=s["block_hit_count"], executed_nonoverlap_hits=s["executed_nonoverlap_hits"],
                            reused_tokens=reused, coverage=reused/len(expected_ids), projection_audit=audit,
                            projection_records=records, correctness=quality,
                            resident_bytes_before=resident_before,
                            resident_bytes_after=semcache.cache.physical_tensor_bytes if semcache else 0,
                            raw_storage_audit=raw_storage_audit(semcache.cache, storage) if semcache else None,
                            cache_capacity_bytes=capacity, cache_fetch_calls=result["cache_fetch_calls"],
                            h2d_cache_bytes=result["h2d_cache_bytes"], memory_before=before, memory_after=after,
                            admission_audit=s.get("admission_audit", []),
                            cache_event_counts={k: sum(e.get("event_type") == k for e in result.get("events", [])) for k in ("INSERT", "EVICT", "DENY")},
                            safe_reuse_claimed=False, use_cache=False, output_attentions=True, attention_implementation="eager")
                        batch.append(row)
                        if extension:
                            extension.decorate(row, variant, semcache, result)
                        if mode == "FULL_RECOMPUTE":
                            for metric in ("native_qkv_host_ms", "native_qkv_gpu_ms"):
                                if metric in row["timing_scopes"]:
                                    row["timing_scopes"][metric]["parent"] = "prefill_gpu_ms" if metric.endswith("_gpu_ms") else "prefill_wall_ms"
                        guard.evidence()
                        del result, semcache
                    del reference
                # All four modes for every request are atomically committed together.
                atomic_json(checkpoint, dict(identity=manifest["identity"], rows=batch))
                rows = [r for r in rows if not r["pair_id"].startswith(name + "-")] + batch
                if extension:
                    extension.write_reports(args.output_dir, rows)
                else:
                    write_reports(args.output_dir, rows)
                print("COMPLETE", name, "rows", len(batch), flush=True)
    manifest.update(status="COMPLETE", raw_only=guard.evidence(), rows=len(rows),
        strict_parity_failures=sum(not r["correctness"]["allclose"] for r in rows
            if r["mode"] in ("FULL_RECOMPUTE", "LOOKUP_NO_REUSE", "POSITION_ALIGNED_DIAGNOSTIC")
            or r["condition"] == "cold_miss"))
    if manifest["strict_parity_failures"]:
        manifest["status"] = "CORRECTNESS_FAILED"
    manifest["strict_nonzero_validated_measured_cases"] = sum(
        r["phase"] == "measured" and r["mode"] == "POSITION_ALIGNED_DIAGNOSTIC"
        and r["condition"] == "same_user_exact" and r["correctness"]["nonzero_reuse_validated"] for r in rows)
    if not manifest["strict_nonzero_validated_measured_cases"] and manifest["status"] == "COMPLETE":
        manifest["status"] = "NO_VALIDATED_REUSE"
    if extension:
        extension.finalize(args, manifest, rows)
    atomic_json(args.output_dir / "manifest.json", manifest)
    return 0 if manifest["status"] == "COMPLETE" else 2


def main(argv=None, extension=None):
    args = parse_args(argv, extension=extension)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("resume", "output_dir")}
    provenance = code_provenance()
    if extension:
        for p in extension.provenance_files(ROOT):
            provenance['files'][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    identity = digest(dict(config=config, files=provenance["files"]))
    path = args.output_dir / "manifest.json"
    if args.resume:
        manifest = json.loads(path.read_text())
        if manifest["identity"] != identity:
            raise SystemExit("Resume rejected: arguments or code hashes changed; use a new output directory")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        manifest = dict(schema_version=1, identity=identity, config=config, code=provenance,
            python=platform.python_version(), platform=platform.platform(), status="INITIALIZING",
            benchmark="controlled microbenchmark; synthetic PEFT users; no safety claim",
            generate_settings="no generate; one prefill; use_cache=False; output_attentions=True; eager",
            timing_policy=dict(request="synchronize before start and before close; correctness outside",
                prefill="nested in request; detailed synchronizes prefill end",
                projection="nested in prefill; cache_fetch/index_transfer/assembly/native nested in mixed_executor",
                cpu="host submission times may overlap GPU; never sum CPU and GPU intervals",
                cache_materialization="nested within cache_policy; source build excluded from target",
                minimal="legacy CPU timers, audit and events retained; no added inner CUDA events; minimal engine prefill omitted",
                pass_comparison="breakdown vs minimal includes timing and detailed-executor control overhead, not just CUDA event cost"))
    manifest["status"] = "RUNNING"
    if extension:
        manifest['experiment'] = extension.name
        extension.initialize_manifest(manifest)
    for old_failure in ("failure_type", "failure", "oom"):
        manifest.pop(old_failure, None)
    atomic_json(path, manifest)
    groups = len(args.lengths) * len(args.storages) * (2 if args.instrumentation == "both" else 1) * (args.warmup_runs + args.measured_runs)
    if extension:
        groups *= len(extension.variants(args))
    manifest["planned_forward_counts_fresh_run"] = dict(target=groups*16, reference=groups*4, source=groups*9,
        note="4 conditions x 4 modes; source only for 3 non-cold conditions x 3 engine modes; resumes rewarm")
    print(f"R1 work: {groups} pairing groups; {groups*16} targets, {groups*4} references, {groups*9} source forwards; smoke={args.smoke}", flush=True)
    try:
        missing = [x for x in ("torch", "transformers", "peft") if importlib.util.find_spec(x) is None]
        if missing:
            raise RuntimeError("NOT EXECUTED: missing dependencies: " + ", ".join(missing))
        with raw_only_guard() as guard:
            return run(args, manifest, guard, extension=extension)
    except Exception as exc:
        manifest.update(status="FAILED", failure_type=type(exc).__name__, failure=str(exc),
                        oom=("out of memory" in str(exc).lower() or isinstance(exc, MemoryError)))
        atomic_json(path, manifest)
        print(f"R1 FAILED ({type(exc).__name__}): {exc}; manifest={path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
