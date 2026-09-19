#!/usr/bin/env python3
"""Tiny opt-in real TinyBERT + OPT-125M structural integration probe."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semcache.models.loader import load_model
from semcache.models.lora_fixtures import create_controlled_users
from semcache.models.model_adapter import OPTModelAdapter
from semcache.semantic.encoder import TinyBERTSemanticEncoder
from semcache.semantic.intent_clusterer import IntentClusterer
from semcache.cache.global_cache import GlobalCache
from semcache.semcache_engine import SemCacheEngine
from semcache.evaluation.mixed_control import validate_mixed_control
from semcache.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/m7/m7_semcache_integration.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--pbr-history-lambda", type=int, default=4,
                        help="Smoke-only REPRODUCTION_CHOICE; paper default is 100")
    args = parser.parse_args()
    seed_everything(42)
    revision = "27dcfa74d334bc871f3234de431e71c6eeba5dd6"
    model, tokenizer, model_metadata = load_model(dict(
        name="facebook/opt-125m", tokenizer="facebook/opt-125m", revision=revision,
        tokenizer_revision=revision, dtype="float32", device=args.device,
        local_files_only=not args.allow_download, attention_implementation="eager"))
    model, lora_metadata = create_controlled_users(model)
    encoder = TinyBERTSemanticEncoder(device=args.device, dtype="float32",
                                      local_files_only=not args.allow_download)
    texts = [
        "Please find a hotel in Cambridge near the station.",
        "Book an Italian restaurant in central London tonight.",
    ]
    warmup = encoder.encode(texts)
    clusterer = IntentClusterer(2, initialization="first_k", update_mode="immediate_eq9")
    # first_k means each warmup embedding is already incorporated once: N_c=1.
    clusterer.initialize(warmup)
    cache = GlobalCache(256 * 1024 * 1024)
    engine = SemCacheEngine(model, tokenizer, OPTModelAdapter(model), encoder, clusterer, cache,
                            rho=.8, history_lambda=args.pbr_history_lambda,
                            metadata=model_metadata, pbr_interval_queries=None)
    control = validate_mixed_control(model, tokenizer, OPTModelAdapter(model), texts[0])
    trace = [("miss", "user_a", texts[0]), ("hit_one", "user_b", texts[0]),
             ("unrelated", "user_a", texts[1]), ("hit_two", "user_b", texts[0])]
    rows = [engine.query(text, user, query_id, compare_baseline=True)["summary"]
            for query_id, user, text in trace]
    pbr = engine.recalculate_cluster(rows[0]["cluster_id"], trigger="forced_smoke_REPRODUCTION_CHOICE")
    events = engine.events
    chu = [event for event in events if event["event_type"] == "CHU"]
    admissions = [event for event in events if event["event_type"] == "INSERT"]
    result = {
        "environment": {**model_metadata, "attention_implementation": "eager"},
        "semantic_encoder": encoder.metadata,
        "clustering": {"cluster_count": 2, "distance_metric": "euclidean_l2",
            "initialization": "first_k_REPRODUCTION_CHOICE",
            "update_mode": "immediate_eq9",
            "centroid_update_rule": "Eq.9 incremental mean after every assignment",
            "assignments": [{"query_id": row["query_id"], "cluster_id": row["cluster_id"],
                             "nearest_centroid_distance_pre_update": row["nearest_centroid_distance_pre_update"],
                             "centroid_update_applied": row["centroid_update_applied"],
                             "cluster_count_before": row["cluster_count_before"],
                             "cluster_count_after": row["cluster_count_after"],
                             "centroid_shift_l2": row["centroid_shift_l2"]} for row in rows],
            "provenance": {"equations": "PAPER_DEFINED", "initialization": "REPRODUCTION_CHOICE"}},
        "reuse": {"query_count": len(rows), "block_lookup_count": sum(r["block_lookup_count"] for r in rows),
            "block_hit_count": sum(r["block_hit_count"] for r in rows),
            "reused_token_count": sum(r["reused_unique_token_count"] for r in rows),
            "physical_reuse_used": any(r["reused_unique_token_count"] for r in rows),
            "projection_skip_used": any(r["reused_projection_rows"] for r in rows)},
        "semantic_impact": {"provider": "actual_attention_probabilities",
            "reducer": engine.impact_reducer.metadata, "rho": .8,
            "lambda": args.pbr_history_lambda,
            "lambda_note": "smoke override; paper default=100; REPRODUCTION_CHOICE",
            "PBR_trigger_policy": "forced_smoke_REPRODUCTION_CHOICE",
            "initial_impact_examples": [event["I"] for event in admissions[:3]],
            "CHU_event_count": len(chu), "PBR_event_count": len(pbr),
            "example_CHU": chu[:1], "example_PBR": pbr[:1],
            "provenance": {"rho_and_lambda_defaults": "PAPER_DEFINED",
                           "reducer_and_trigger": "REPRODUCTION_CHOICE"}},
        "cache": {"admission_count": len(admissions),
            "eviction_count": sum(e["event_type"] == "EVICT" for e in events),
            "final_logical_bytes": cache.logical_cache_bytes,
            "peak_logical_bytes": cache.peak_logical_cache_bytes,
            "selected_cache_entry_diagnostics": [engine.entry_fields(e) for e in list(cache.entries.values())[:3]]},
        "correctness": {"controlled_exact_reuse_projection_error": control["projection_max_abs_error"],
            "controlled_exact_reuse_logit_error": control["max_abs_logit_diff"],
            "argmax_agreement": control["last_argmax_agreement"], "safe_reuse_claimed": False},
        "lora": lora_metadata, "rows": rows,
        "scope": "small structural integration; not paper numerical reproduction"
    }
    required = (admissions and chu and pbr and result["reuse"]["physical_reuse_used"]
                and result["reuse"]["projection_skip_used"])
    if not required:
        raise AssertionError("M7 probe did not exercise admission, hit/CHU, PBR and physical skipping")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"cluster={rows[0]['cluster_id']} admissions={len(admissions)} hits={result['reuse']['block_hit_count']} ")
    print(f"CHU={len(chu)} PBR={len(pbr)} physical_reuse={result['reuse']['physical_reuse_used']} ")
    print(f"projection_skip={result['reuse']['projection_skip_used']} safe_reuse_claimed=false")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
