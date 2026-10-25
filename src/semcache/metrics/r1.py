"""R1 artifact and audit utilities. Importable without torch or model packages."""
from collections import defaultdict
from contextlib import contextmanager
import csv
import hashlib
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
from statistics import mean
import sys

MODES = ("FULL_RECOMPUTE", "LOOKUP_NO_REUSE", "RAW_PHYSICAL_REUSE",
         "POSITION_ALIGNED_DIAGNOSTIC")
ENGINE_MODES = dict(zip(MODES[1:], ("SEMCACHE_LOOKUP_NO_REUSE",
    "SEMCACHE_PHYSICAL_REUSE", "SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC")))
FORBIDDEN = ("semcache.experiments.cachegen", "cachegen", "CacheGen", "bitsandbytes")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class RawOnlyGuard(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Fail closed on imports of the repository's compression runtime or codecs.

    Not a sandbox: the runner also rejects quantized models and fingerprints its
    imported local code. No codec dependency is constructed by R1.
    """
    def __init__(self):
        self.blocked_import_attempts = 0

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in FORBIDDEN):
            # HF probes optional packages with find_spec; reject execution, not
            # harmless availability probing. Nothing from the module is loaded.
            original = importlib.machinery.PathFinder.find_spec(fullname, path)
            if original is None:
                return None
            return importlib.machinery.ModuleSpec(fullname, self, origin=original.origin,
                is_package=original.submodule_search_locations is not None)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.blocked_import_attempts += 1
        raise RuntimeError("R1 forbids compression runtime: " + module.__name__)

    def evidence(self):
        loaded = [n for n in sys.modules if any(n == p or n.startswith(p + ".") for p in FORBIDDEN)]
        if loaded or self.blocked_import_attempts:
            raise RuntimeError(f"Raw-only guard violated: {loaded}, attempts={self.blocked_import_attempts}")
        return dict(forbidden_module_prefixes=FORBIDDEN, loaded_forbidden_modules=loaded,
                    blocked_import_attempts=self.blocked_import_attempts,
                    codec_calls=0, quantization_enabled=False, storage_compression=False,
                    transport_compression=False,
                    evidence_kind="blocked import paths + raw tensor ownership + unquantized projection validation")


@contextmanager
def raw_only_guard():
    guard = RawOnlyGuard()
    guard.evidence()
    sys.meta_path.insert(0, guard)
    try:
        yield guard
        guard.evidence()
    finally:
        sys.meta_path.remove(guard)


def audit_projection_records(records, tokens, layers, reused, reused_positions=None):
    """Validate executed row evidence, not just a logical cache hit count."""
    if len(records) != 3 * layers:
        raise AssertionError("Missing projection execution records")
    if {(r["layer"], r["tensor_type"]) for r in records} != {(i,n) for i in range(layers) for n in "qkv"}:
        raise AssertionError("Duplicate or missing layer/role identity")
    if reused_positions is not None and len(set(reused_positions)) != reused:
        raise AssertionError("Reused coverage does not match executed spans")
    total_native = total_skipped = 0
    for record in records:
        positions = record["native_positions"]
        if reused_positions is not None and positions != [i for i in range(tokens) if i not in set(reused_positions)]:
            raise AssertionError("Native projection consumed cached positions")
        if (len(set(positions)) != len(positions) or any(p < 0 or p >= tokens for p in positions)
                or record["native_projection_rows"] != tokens - reused
                or record["reused_projection_rows"] != reused
                or len(positions) != tokens - reused
                or record["native_call_count"] != int(tokens > reused)):
            raise AssertionError("Projection row accounting mismatch")
        total_native += record["native_projection_rows"]
        total_skipped += record["reused_projection_rows"]
    return dict(native_projection_rows_all_layers_roles=total_native,
                skipped_projection_rows_all_layers_roles=total_skipped,
                projection_rows_per_role_per_layer=reused,
                executed_projection_records=len(records))


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values)-1)] - values[lower]) * (position-lower)


def statistics(values):
    return dict(mean=mean(values), p50=percentile(values, .5), p95=percentile(values, .95)) if values else dict(mean=None, p50=None, p95=None)


def summarize(rows):
    measured = [r for r in rows if r["phase"] == "measured"]
    keys = ("storage", "instrumentation", "requested_prompt_tokens", "condition", "mode")
    groups = defaultdict(list)
    full = {r["pair_id"]: r for r in measured if r["mode"] == "FULL_RECOMPUTE"}
    for row in measured:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        result = dict(zip(keys, key), count=len(group), diagnostic="DIAGNOSTIC" in key[-1])
        names = sorted({n for r in group for n, v in r["timing"].items() if n.endswith("_ms") and isinstance(v, (int, float))})
        for name in names:
            for stat, value in statistics([r["timing"][name] for r in group if r["timing"].get(name) is not None]).items():
                result[f"{name}_{stat}"] = value
        deltas = [r["timing"]["request_wall_ms"] - full[r["pair_id"]]["timing"]["request_wall_ms"] for r in group]
        result.update({"matched_delta_ms_" + k: v for k, v in statistics(deltas).items()})
        result.update(mean_coverage=mean(r["coverage"] for r in group),
                      parity_passes=sum(r["correctness"]["allclose"] for r in group),
                      reuse_validated_cases=sum(r["coverage"] > 0 and r["correctness"]["allclose"] for r in group),
                      safe_reuse_claimed=False)
        output.append(result)
    return output


def comparison_checks(rows):
    """Require equal source residency and selected spans across storage/passes."""
    signatures = {}
    for row in rows:
        key = (row["phase"], row["repeat"], row["requested_prompt_tokens"], row["condition"], row["mode"])
        value = (row["source_signature"], row["reuse_signature"], row["token_ids_sha256"])
        if key in signatures and signatures[key] != value:
            raise AssertionError(f"Storage/instrumentation changed source residency or selected hits: {key}")
        signatures[key] = value
    return True


def timing_scopes(timing):
    """Machine-readable nesting; intentionally no additive total reconstruction."""
    result = {}
    for name in timing:
        if not name.endswith("_ms"):
            continue
        parent, kind = "request_wall_ms", "inclusive"
        if name == "request_wall_ms":
            parent = None
        elif name == "cache_materialization_ms":
            parent = "cache_policy_ms"
        elif name == "prefill_gpu_ms":
            parent = "prefill_wall_ms"
        elif name.startswith(("native_qkv_", "cache_fetch_", "tensor_assembly_", "index_transfer_")):
            parent = "mixed_executor_gpu_ms" if name.endswith("_gpu_ms") else "mixed_executor_host_ms"
        elif name.startswith("mixed_executor_") or name in ("mixed_qkv_execution_ms", "qkv_execution_ms"):
            parent = "prefill_gpu_ms" if name.endswith("_gpu_ms") or name.endswith("execution_ms") else "prefill_wall_ms"
        elif name in ("quality_diagnostics_ms",):
            parent, kind = None, "outside_request"
        elif name == "prefill_host_overhead_ms":
            parent, kind = "prefill_wall_ms", "derived_do_not_sum"
        result[name] = dict(parent=parent, kind=kind, additive_to_parent=False,
            clock="cuda_stream_event" if name.endswith("_gpu_ms") or name in ("mixed_qkv_execution_ms", "qkv_execution_ms") else "cpu_wall",
            available=timing[name] is not None)
    return result


def write_reports(root, rows):
    root = Path(root)
    comparison_checks(rows)
    (root / "raw.jsonl").write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows))
    summary = summarize(rows)
    fields = sorted({k for row in summary for k in row})
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(summary)
    atomic_json(root / "correctness.json", [dict(pair_id=r["pair_id"], mode=r["mode"],
        phase=r["phase"], coverage=r["coverage"], **r["correctness"]) for r in rows])
    pairs = {(r["storage"], r["requested_prompt_tokens"], r["condition"], r["mode"], r["repeat"]): r
             for r in rows if r["phase"] == "measured" and r["instrumentation"] == "minimal"}
    overhead = []
    for r in rows:
        key = (r["storage"], r["requested_prompt_tokens"], r["condition"], r["mode"], r["repeat"])
        if r["phase"] == "measured" and r["instrumentation"] == "breakdown" and key in pairs:
            overhead.append(dict(storage=key[0], length=key[1], condition=key[2], mode=key[3], repeat=key[4],
                breakdown_minus_minimal_ms=r["timing"]["request_wall_ms"]-pairs[key]["timing"]["request_wall_ms"]))
    atomic_json(root / "instrumentation_overhead.json", overhead)
    lines = ["# R1 measured prefill baseline", "", "No safety claim. Diagnostics are separate; no generation or KV-prefix reuse.",
             "Nested GPU intervals are not additive to prefill/request. Failed parity is not validated acceleration.", "",
             "|Storage|Pass|Length|Case|Mode|n|Request mean ms|p50|p95|Paired delta ms|Coverage|Parity passes|",
             "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append("|" + "|".join(str(r[k]) for k in ("storage", "instrumentation", "requested_prompt_tokens", "condition", "mode", "count", "request_wall_ms_mean", "request_wall_ms_p50", "request_wall_ms_p95", "matched_delta_ms_mean", "mean_coverage", "parity_passes")) + "|")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
