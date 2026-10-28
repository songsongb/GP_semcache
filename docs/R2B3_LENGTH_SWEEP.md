# R2-B3 Exact Prefix KV length sweep

Initial checkout: `exp/raw-semcache-r2`, HEAD
`ac9727f87e6080b181f91d07d3a36eb7941526b9`, clean working tree. B1/B1.5/B2
implementations and documentation were inspected. The ignored local B2 result
directories contain two FAILED dependency-check manifests (missing Torch), with
GPU_SMOKE_VALIDATED=false; user-supplied GPU findings remain historical
evidence, never new B3 samples. Frozen code/results and Slurm files are preserved.

Status: **IMPLEMENTED_LOCAL_ONLY; GPU validation NOT_RUN**. No measured B3 GPU
speedup, capacity, numerical improvement or break-even is claimed.

## Workloads and identity

| Group | L | P | S |
|---|---|---|---|
| A | 128,256,512,1024 | L−64 | 64 |
| B | 128,256,512,1024 | L−1 | 1 |
| C no-hit | 128,512,1024 | 0 | L |
| C middle | 512,1024 | 256,768 | 256 |
| Smoke | 32 | 16,31 | 16,1 |

Default pilot is five representative cases: 128:64,256:192,512:448,1024:960,
1024:1023. `--case-group AB` includes eight conditions. Explicit `--cases L:P,...`
overrides groups. Maximum L=1024 and actual model positional limits are enforced;
P+S=L, 0≤P<L, S≥1. No 2048+ path or Cartesian sweep.

Uses existing deterministic **per-length** tokenizer fixture and exact roundtrip.
This is not a shared nested-length sequence design: no assumption that one
length's tokens prefix another's. Actual IDs, length and hashes are stored per
case. The same IDs/weights/adapter/backend/source snapshot are bound across
paired modes and validated against pilot evidence. P/L is token-prefix coverage,
not a measured fraction of FLOPs or GPU compute saved.

## Frozen execution, clocks and quality

The new runner imports `B2Runtime`, `run_pairs`, mode scheduling, masks/positions,
immutable snapshot and single-use Prepared machinery. No forward executor body
is added or edited. Full processes L, Normal/Prepared process S in all decoder,
QKV and FFN layers; output KV length is L. Source KV comes only from independent
P-token prefill; target Full cache never populates a source snapshot.

Normal includes namespace/shape checks, KV clones/container, input/mask/position
creation, adapter/eval and native forward inside the B2 synchronized CPU request
wall. Source prefill and canonical snapshot costs remain outside and are reported
separately. CPU Logit transfer, cache/hash audits, hooks, JSON/CSV, and frame
cleanup retain B2's outside-timer boundary. This is device-logits/KV latency,
not generation TTFT or full host-response latency.

Prepared is fresh, one-use `IDEAL_DIAGNOSTIC_ONLY`, with only preparation moved
outside timer and the original guards retained. Never primary speedup evidence.
No-hit P=0 has no source or Prepared; null fields mean NOT_APPLICABLE.
Normal/Prepared Logits **and** final KV must match exactly. Canonical hashes,
positions/causal masks, namespace and final model weights are verified. Structural
or cache/parity failures invalidate the case and abort execution. Strict FP16
FAIL alone allows diagnostic timing and is preserved in every row/decision.

B3 explicitly applies `abs(actual-reference) <= 1e-3 + 1e-3*abs(reference)` to
the entire aligned suffix and separately to next-token logits. Existing numeric
helpers use the other argument as relative-tolerance reference in some metrics;
B3 retains their descriptive values and `legacy_b2_allclose`, but computes its
strict decision with the requested reference-relative predicate. A scoped
callback changes only untimed correctness calculation while reusing B2 pairing.
Frozen helpers/results are untouched. Double precision is used to evaluate
already-computed Logit differences; model inference precision is unchanged.
Argmax agreement never replaces strict equality requirements. The inherited B1
namespace retains loader-default output-attention metadata; B3's explicit arm,
condition identity and actual forward-output audit are authoritative. Every arm
constructs its own source; cross-arm cache sharing is not performed.

## Endpoint and attention

Only `--endpoint-policy b1` is supported. Outputs remain Full [1,L,V] and Warm
[1,S,V], final KV length L. Every condition records `b1_all_logits`,
`output_matched=false`, `materialization_confounded=true`, and
`B1_COMPATIBLE_LATENCY_OBSERVATION`. A lower Normal wall time under this endpoint
is not isolated evidence of Prefix-only computation benefit or an output-matched
service speedup. Custom decoder/head control is
**OUTPUT_MATCHED_CONTROL_UNAVAILABLE**: no custom path has been validated for
the installed API/dtypes/lengths, and B2's reported native API lacks the option.
It is deliberately not introduced into the primary baseline.

Default arm is attention OFF with eager/use_cache=True. ON is independently
audited, never pooled with OFF. The B2 backend class/config/source and actual
returned-attention checks are reused; unsupported/fallback implementations fail
explicitly. OFF does not imply eager avoids internal probability computation.
Pilot runs OFF cases first, ON second if requested. Later case order is seeded
and shuffled, while paired mode ordering retains B2's counterbalanced schedule.

## Capacity approval and failure accounting

Pilot defaults to one warmup and three paired samples, plus untimed structural
proofs. Its timing is tagged `benchmark_phase=pilot` and rejected by aggregate.
CUDA peaks are reset **outside** timers before source construction; snapshots
include source/build/gate/request peak allocated/reserved memory. Host RSS and
process high-water RSS are read from /proc where available, otherwise null.
These are case/process memory observations, not per-operation allocator costs.

Original B1 memory screen is conservative and unchanged, even for attention OFF.
It can reject a configuration before an actual OOM; rejection is explicitly
MEMORY_PREFLIGHT_FAILED, never asserted as an empirical hardware limit. It uses
large attention/workspace reserves and may reject 1024 on a 24GB card. No lower
length is silently substituted and no screen is relaxed.

OOM/preflight failures retain length/phase/traceback and memory-at-failure.
They permit subsequent capacity cases. Cleanup, synchronization and optional
empty_cache occur only at that failed-case boundary, outside request timers.
Other runtime/safety failures abort. COMPLETE_WITH_FAILURES differs from COMPLETE;
failed-case partial samples remain raw evidence and are excluded from aggregation.

Latency and profile require `--pilot-inputs` from completed B3 pilot runs with
final weight checks. Every selected exact L/P/arm requires CAPACITY_VALIDATED
under matching code, model/tokenizer/adapter weights, API/software, device UUID,
seed and endpoint. The entire selected plan is checked before any target/source
request. After source preparation, actual token/cache condition identity must
also match. Changed namespaces, workloads, hardware or stale code cannot inherit
approval. Run a pilot for optional C/ON cases separately. If 1024 fails, explicitly
select only validated lengths; preserve the failed pilot artifacts.

## Per-request and source accounting

Raw pairs contain L/P/S, prefix/suffix ratios, cache hit/bytes, source prefill and
snapshot costs, Normal cache-prepare host enqueue interval/clone bytes, Full,
Normal and Prepared walls/differences/ratio, full quality metrics, row audit,
model/dtype/adapter and source identity, attention/backend/endpoint, memory,
session/pair/order IDs. No missing timing is encoded as zero.

One separate PREPARE_ONLY trial after paired requests records synchronized
materialize wall in source_costs. Raw `prepare_wall_ms` is null, with a pointer
status to that separate component trial, not a copied per-pair estimate. Normal's
in-request `normal_cache_prepare_host_ms` is asynchronous enqueue/container wall;
completion is included in the enclosing request synchronization. Never add this
or standalone preparation to Normal wall. No source amortization simulator.

## Analysis and break-even labels

Only B3 latency manifests are accepted; smoke, pilot and profiler data are rejected.
Each VALID condition needs ≥20 complete measured pairs and matching model/case/
backend/condition identity. Duplicate sessions/processes, hardware UUID mismatch,
code/API/namespace mismatch and incomplete evidence are rejected. Failed cases
and missing sessions are preserved as BREAK_EVEN_UNRESOLVED.

Within each session, Δ=Full−Normal and ratio=Full/Normal. Across processes, session
means receive equal weight. Report mean/p50/p95/std, paired wins, per-session and
between-session variance, retained outliers and exploratory 95% bootstrap CI.
Bootstrap resamples independent session means; no pooled-pair independent CI.
With three sessions the CI has limited resolution and confidence.

- NO_OBSERVED_SPEEDUP: nonpositive mean under verified conditions.
- POSITIVE_MEAN_ONLY: positive mean without required reproduced support.
- REPRODUCED_POSITIVE_LATENCY_DIFFERENCE: ≥3 sessions, all positive means,
  positive exploratory CI and verified environment, with no missing sessions.
- BREAK_EVEN_REGION_CANDIDATE: adjacent measured doubled lengths in the same
  fixed-S/attention family both reproduce a positive difference. Only endpoints
  were measured; interval interiors are not asserted.
- BREAK_EVEN_UNRESOLVED: invalid/missing evidence, unverified environment, no-hit.
- CORRECTNESS_QUALIFIED_SPEEDUP is reserved for appropriate output-matched strict
  evidence; B3's confounded primary endpoint always leaves its Boolean false.

No numeric threshold interpolation or extrapolation is performed. Amortized
break-even is deferred to B4. Analyze records analysis-process environment
separately, derives model/dtype/cases/arms from input manifests and fingerprints
the inputs. Analysis defaults cannot be mistaken for experimental configuration.

## CLI (SERAPH, offline)

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/88_run_r2b3_length_sweep.py --phase smoke \
  --model facebook/opt-125m --dtype float32 --cases 32:16,32:31 \
  --device cuda:0 --cache-device cuda:0 --attention-arm off \
  --output-dir results/r2/b3/smoke01

# First capacity phase: representative five cases (default group=pilot).
python scripts/88_run_r2b3_length_sweep.py --phase pilot \
  --model facebook/opt-2.7b --dtype float16 --attention-arm off \
  --device cuda:0 --cache-device cuda:0 --warmup-runs 1 --measured-runs 3 \
  --independent-run-id pilot01 --output-dir results/r2/b3/pilot01

# Additional three B cases, only after inspecting the representative pilot.
python scripts/88_run_r2b3_length_sweep.py --phase pilot \
  --model facebook/opt-2.7b --dtype float16 --attention-arm off \
  --cases 128:127,256:255,512:511 --device cuda:0 --cache-device cuda:0 \
  --independent-run-id pilot_b01 --output-dir results/r2/b3/pilot_b01

# Execute only if ALL selected A/B conditions passed capacity pilots.
for run in 1 2 3; do
  python scripts/88_run_r2b3_length_sweep.py --phase latency \
    --model facebook/opt-2.7b --dtype float16 --case-group AB --attention-arm off \
    --device cuda:0 --cache-device cuda:0 --endpoint-policy b1 \
    --pilot-inputs results/r2/b3/pilot01 results/r2/b3/pilot_b01 \
    --warmup-runs 3 --measured-runs 20 --seed 42 --order-seed "$((1100 + run))" \
    --independent-run-id "run${run}" --output-dir "results/r2/b3/latency_run${run}"
done

python scripts/88_run_r2b3_length_sweep.py --phase analyze \
  --inputs results/r2/b3/latency_run1 results/r2/b3/latency_run2 results/r2/b3/latency_run3 \
  --order-seed 901 --output-dir results/r2/b3/aggregate01
```

If 1024 was rejected, replace `--case-group AB` identically in all three commands
with `--cases 128:64,256:192,512:448,128:127,256:255,512:511`, provided those all
passed. For ON or C, first run new matching pilots. Never reuse output directories.
Pin `--revision`/`--tokenizer-revision` consistently for exact reproduction.

Optional targeted profile after sweep analysis:

```bash
python scripts/88_run_r2b3_length_sweep.py --phase profile \
  --model facebook/opt-2.7b --dtype float16 --cases 128:127,512:511 \
  --attention-arm off --device cuda:0 --cache-device cuda:0 \
  --pilot-inputs results/r2/b3/pilot01 results/r2/b3/pilot_b01 \
  --output-dir results/r2/b3/profile01
```

This reuses B2's one-request CPU/CUDA profiler/ranges/parity/trace parser and
separate component clocks. It never feeds profiler latency into minimal stats.
Kernel-duration sums, CUDA intervals and CPU wall are separate; parent/child
times are never added. Specific kernel mechanisms/CPU dominance remain unresolved.
No broad auto-profile or Slurm file is generated.

## Artifacts and local validation

Run artifacts: manifest/environment/workload_config, correctness_and_gates,
cache_integrity, forward_skip_audit, attention_backend_audit, endpoint_parity,
source_costs and decision_summary JSON. Pilot adds capacity_pilot.json. Minimal
phases add latency_raw_pairs.csv/.json and latency_session_summary.csv. Analyze
adds latency_aggregate_summary.json, break_even_summary.csv and
break_even_decision.json. Profile uses B2 profiling CSV/trace artifacts. Global
failures retain manifest/decision traceback; completed pair checkpoints remain.
No missing model-run measurements are fabricated when dependencies are absent.

```bash
PYTHONPATH=src:tests python3 -m unittest test_r2b3_workloads test_r2b3_capacity \
  test_r2b3_analysis test_r2b3_benchmark test_r2b2_benchmark \
  test_r2b2_attention_ablation test_r2b2_analysis test_r2b15_numeric \
  test_r2b15_cost test_r2b1_prefix_cache test_r2b1_prefix_runtime -q
python3 scripts/88_run_r2b3_length_sweep.py --help
git diff --check
```

CPU scalar/control tests cover reference-relative threshold, frozen callback
restoration, canonical/Prepared mutation rejection, pilot approvals and partial
OOM artifacts, process-level aggregation, outliers, endpoint and manifest/hardware
mismatch. Tiny native model tests are dependency/CUDA gated; synthetic statistics
are test data, never performance evidence.

Local verification (2026-10-28): 100 tests total across B3/B2/B1.5/B1/R2-A,
81 passed, 19 skipped, zero failures. New B3 suite: 20 tests, 18 passed and two
model/CUDA dependency skips. Syntax compilation, CLI help and command argument
parsing passed. Torch/Transformers/PEFT are unavailable locally; actual GPU
capacity, FP16 accuracy, profile and break-even remain NOT_RUN/UNVERIFIED.
