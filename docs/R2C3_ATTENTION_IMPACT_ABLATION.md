# R2-C3 Attention Impact conditional-request ablation

Checkout inspected: `exp/raw-semcache-r2`, HEAD
`cdbf273800c24398313bc7b578a49ebc25504d2b`, initially clean. Inspected 90,
C1 runtime/analysis, C2 memory guard, frozen B3/R2-A runtime, Engine, impact
execution, history, CHU, Admission/Eviction and source clone paths. Historical
C2 GPU artifacts and `jobs/r2c3_samegpu_ablation.sh` are **not present locally**.
The reported C1/C2 timings/kernel durations are historical user evidence, not
new measurements. No historical Python/result or shell/Slurm file is modified.

## Question and five arms

Does excluding actual impact reduction and its scalar materialization change
the existing synchronized **conditional-request** wall clock, with exact policy
inputs and decisions preserved?

| Identifier | Existing path | Impact |
|---|---|---|
| FULL_NATIVE | Frozen C1 native Full, native all-token head | No Engine impact |
| SEMCACHE_NO_REUSE_IMPACT_ON | C1 lookup/Engine, no physical reuse | Actual token_precompute |
| SEMCACHE_NO_REUSE_IMPACT_REPLAY | Same No-Reuse path | Exact mode-specific reference |
| RAW_SEMCACHE_QKV_IMPACT_ON | C1 fast position-aligned Catalog + contiguous/fallback executor | Actual token_precompute |
| RAW_SEMCACHE_QKV_IMPACT_REPLAY | Same selected QKV and executor | Exact mode-specific reference |

All use the same single OPT model and synthetic rank-8 `user_a` fixture, eager
attention, use_cache=False, output_attentions=True, and `[1,L,V]` native logits.
Full bypasses the Engine. No-Reuse retains its original exact-w3 logical lookup
and non-overlap control; RAW uses the existing fast occurrence selector. Actual
semantic encoder/cluster/policy operations remain. Even at requested coverage 0,
RAW remains the RAW wrapper with zero physical skip. No-Reuse logical hits are
not relabeled cache misses. Source QKV stays resident across the compared arms.

This is **ORACLE_DIAGNOSTIC / ORACLE_UPPER_BOUND**, not a deployable impact
predictor. safe_reuse_claimed=false, production_speedup_claimed=false and
correctness_qualified_speedup=false remain false even if every strict check passes.
It is not Native Prefix KV reuse or production policy integration.

## Exact reference and scoped Replay

Each exact L/coverage/source condition creates **two independent untimed ON
references**, one per SemCache base mode. Each captures all `L-2` sliding w=3
candidates, including duplicates and overlaps, in actual extraction order. It
does not capture only selected reuse windows. References contain actual Python
impact values and their float.hex representations, candidate IDs/positions,
attention shape/dtype/device metadata, valid mask, model/base+adapter weight and
namespace identity, source snapshot/backing digest, code hash, requested coverage,
storage/capacity and mode. A sealed immutable reference digest is checked before
every use. A missing reference cannot trigger reference generation during Replay.

During the target request a scoped patch replaces only the actual
`cache.impact_execution.compute_window_values` dispatch. ON delegates to the
original TOKEN_PRECOMPUTE function. Replay validates exact incoming candidate
order/length/Token IDs and attention/mask metadata, then returns the corresponding
precomputed values. It never reads attention tensor contents, moves attention to
CPU, or calls the original reduction. Both arms observe CPU-only metadata and
the existing CPU mask; no attention tensor lifetime is extended.

Counters count **Python reduction dispatches**, not CUDA kernels: ON must perform
exactly one real TOKEN_PRECOMPUTE call; Replay exactly one dispatch with **zero
real reduction calls**. Legacy/fallback reducers and the low-level attention
validation entry are guarded against unexpected Replay computation. Any bypass
or repeated dispatch fails. ExitStack restores every temporary function patch
after success, Forward failure, policy failure or interruption. Execution is
single-threaded; these global scoped patches are not a concurrent serving API.

Excluded from Replay is the original reduction's device indexing/masks and
conversions, finite/L2/head reductions, GPU window aggregation, packed finite
flag/impact scalar CPU materialization. The model **still computes and returns
normal attention tensors**. Semantic encoder CPU materialization, original logit
transfer, and unrelated synchronization are not removed.

The original Engine then performs the same candidate loop, key grouping and
per-key duplicate means, history append, CHU, Admission/Eviction/materialization
and scheduled PBR. Values are never zeroed, predicted, borrowed from another
mode/coverage, or supplied to a different policy implementation.

Reference costs are separate: `reference_conditional_request_ms`,
`reference_request_wrapper_wall_ms` and `reference_generation_wall_ms` (including
reference packing/digest). Untimed physical gate wall time and source prefill,
Catalog/index/descriptor costs are recorded separately. None is silently charged
to or amortized into measured Replay requests.

## Policy and correctness gates

A scoped observer around the real `r1_runtime.engine_request` captures policy
state before the Engine opens its clock and after that clock has closed. It does
not clone a second policy algorithm or modify a policy decision.

State includes target cache keys/metadata/size/impact/frequency/clock, lookup
counters, admission/eviction parameters, appearance counters/window, complete
impact history, updater configuration, cluster counts/centroids/pending updates,
admission order, and **Catalog entry state**. The latter matters because RAW
CHU updates selected sidecar occurrences, not only target Engine cache entries.
Original Catalog policy reset/Engine clone behavior is reused before each request.

Wall timestamps and timing-only fields are excluded. Logical `cache.now`, entry
created_at/last_access/updated_at and history order are retained: they affect age
and policy semantics. Floats compare through exact hexadecimal representation;
there is no policy tolerance. Ordered impact inputs, history/state digests,
CHU/Admission/Deny/Insert/Evict/PBR events, decision counts and configuration must
match the mode's reference and its paired ON/Replay request. Divergence stops the
case and preserves expected/actual proof digests and exception details.

Untimed frozen C1 gates verify source capture/backing equality, token/position/
prefix/adapter alignment, actual projection rows, native all-token head and LoRA
participation, and legacy/contiguous QKV and Logit parity. Reference and Replay
also receive untimed root/head/LoRA audits. Before/after content hashes and
parameter stamps guard canonical Source and model weights. Code is checked before
and after each case and before final completion.

Every paired observation independently checks ON/Replay exact logits with
torch.equal, strict numerical metrics, selection/source provenance, actual
coverage/rows and policy inputs/state. Full-relative metrics on all-token logits
remain separate, using abs(actual-reference) <= 1e-3 + 1e-3*abs(reference).
An FP16 RAW-vs-Full **STRICT_FAIL** can coexist with exact ON/Replay parity and a
completed diagnostic. It never becomes correctness-qualified service speedup.

## Timing boundaries

| Scope | Inside primary clock | Outside primary clock |
|---|---|---|
| Full | Tokenization, adapter/eval, input creation, native Forward/head, boundary sync | Source, CPU logits export, gates/serialization |
| ON | Actual semantic encoding, cluster/lookup/selection, mapping/snapshot checks as executed, setup, Forward/head, real impact, history/policy, boundary sync | Original Engine deepcopy/Catalog reset/controller construction, source, reference generation, proofs/CPU logits/serialization |
| Replay | Same ON path including attention outputs and policy; precomputed dispatch replaces reduction | Same exclusions; no reference generation inside target |

ON/Replay have thin dispatch/CPU-metadata counters. Replay's window/metadata
validation remains inside its measured request. Heavy state snapshots, hashes,
strict Logit comparisons and policy proof construction are outside that clock.
Thus this controlled wall difference includes these disclosed small dispatch
costs; it is not an exact CUDA arithmetic saving.
Removing the packed impact-to-host transfer also removes its host barrier and
can change CPU/GPU overlap with later policy work. The paired wall difference
therefore need not equal the separately profiled impact kernel-duration sum.

`wrapper_wall_ms` retains C1's broader invocation clock, which includes Engine
cloning and the new policy observer's pre/post work, but excludes some C3 controller
checks surrounding that invocation. It is not a serving E2E clock. No profiler
is installed in the minimal loop. C2 kernel sums can overlap/include ranges
outside this clock and are never subtracted to manufacture CPU time. A historical
123.7 ms kernel total does not predict an equal wall-clock reduction.

## Capacity and compatibility

The same conservative C1 estimator (profile=False), actual source QKV capacity
check and inherited R1 source preflight run unchanged. There is no C2 guarded
profile override in C3 and no silent size/dtype/attention/capacity fallback. Pilot
uses the existing reset-aware C2 memory observer outside latency phases. Actual
OOM, preflight rejection, missing evidence and runtime/parity failures preserve
requested workload, memory evidence and full traceback; no case is silently retried.
The existing semantic encoder still has its separate 512-token cap, explicitly
recorded. OPT input IDs are not truncated.

Latency requires a new completed C3 Pilot on the actual same Torch GPU UUID,
model/adapter weights, code/implementation, encoder/revisions, settings, seed,
storage/capacity and exact length/coverage. Condition source identity and newly
captured reference digests must also match. C1/C2 Pilot approvals are not accepted
as C3 approvals. Historical JSON is never edited. Any incomplete Pilot is rejected.
Boundary monitoring rejects observed competing GPU processes; unavailable telemetry
is recorded rather than inventing stability or zero utilization.

## CLI contract and SERAPH workflow

Entry: `scripts/92_run_r2c3_impact_ablation.py`, phases pilot/latency/analyze.
All GPU arguments in the task/launcher contract are accepted without new required
options. Downloads are disabled; optional model/tokenizer/encoder revision flags
can pin offline artifacts. Default OPT-2.7B FP16 lengths512,1024 and coverages0,.5,max
produce exactly the six named condition keys. Native OPT-125M FP32 32/64 lengths
are available only as explicitly selected small Pilot fixtures, not a new smoke
phase or a substitute for the primary Pilot.

The separately authored shell launcher is not in this local checkout. Its exact
body cannot be inspected here; CLI compatibility is checked against the supplied
flag and artifact contract. No `.sh`/`.sbatch` file is created or edited. On SERAPH,
after that separately supplied file is present:

```bash
cd /data/khuss/repos/GP_semcache
sbatch jobs/r2c3_samegpu_ablation.sh
```

Equivalent Python stages, all on one allocated GPU and in separate processes:

```bash
conda activate semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/92_run_r2c3_impact_ablation.py --phase pilot \
  --model facebook/opt-2.7b --dtype float16 --device cuda:0 --cache-device cuda \
  --prompt-lengths 512,1024 --coverages 0,.5,max --capacity-mib 512 \
  --seed 42 --order-seed 4100 --independent-run-id pilot01 \
  --warmup-runs 1 --measured-runs 3 --output-dir results/r2/c3/pilot01

# Only after all Pilot gates pass; do not run processes concurrently.
for run in 1 2 3; do
  python scripts/92_run_r2c3_impact_ablation.py --phase latency \
    --model facebook/opt-2.7b --dtype float16 --device cuda:0 --cache-device cuda \
    --prompt-lengths 512,1024 --coverages 0,.5,max --capacity-mib 512 \
    --seed 42 --order-seed "$((4100 + run))" --independent-run-id "run${run}" \
    --warmup-runs 3 --measured-runs 20 --pilot-inputs results/r2/c3/pilot01 \
    --output-dir "results/r2/c3/latency_run${run}"
done

python scripts/92_run_r2c3_impact_ablation.py --phase analyze \
  --inputs results/r2/c3/latency_run1 results/r2/c3/latency_run2 results/r2/c3/latency_run3 \
  --output-dir results/r2/c3/aggregate01
```

Each output directory must be new. The five arms use deterministic cyclic Latin
blocks with seeded permutations (20 repeats gives each arm each position four
times). Pilot case order is length-ascending; latency case order is shuffled by
recorded order seed. Input tokens/source generation seed stay identical. Pilot
data are not pooled with latency data.

## Artifacts and analysis

GPU phases produce manifest.json with all requested condition keys,
environment.json, timing_boundary_audit.json, workload_audit.json,
correctness_and_gates.json, impact_references.json, ablation_raw_pairs.json,
ablation_session_summary.csv and ablation_decision.json. Pilot additionally
produces capacity_pilot.json. Only fully verified success is COMPLETE with
GPU_VALIDATION=EXECUTED. Accepted Pilot conditions are CAPACITY_VALIDATED; latency
conditions VALID with measured_pairs=20 by default, ablation_equivalence_pass=true
and impact_replay_reduction_calls=0. Each condition includes actual rows/coverage,
source checks, reference provenance/digests and exact policy proof. Missing fields
are never interpreted as successful verification or zero measurements.

Raw rows preserve all five walls, wrapper clocks, order, warmup/measured phase,
dispatch counts, policy/reference/selected-window digests, actual reused/native
rows, mapping/setup statistics, exact parity and Full-relative metrics. Full
selected-window provenance is stored once per condition, not repeated in every
row. No QKV/attention Tensor dump is written. The reference artifact is sealed by
SHA256; analysis rejects missing/altered reference files.

Analyze is CPU-only and requires exactly three **COMPLETE** latency directories
from unique process/run IDs and distinct order seeds. It rejects incompatible GPU/model/weights/code/
endpoint/settings/case sets, incomplete/duplicate samples, missing policy/reference
proof, nonzero Replay reductions and invalid structural gates. It verifies all
warmups and measured pairs and never reinterprets FP16 Strict FAIL as PASS.
Configuration is derived from input manifests; no analysis argparse defaults are
copied into GPU experiment metadata.

Analyze produces manifest.json, ablation_aggregate_summary.json (list of condition
records) and ablation_decision.json. Manifest GPU_VALIDATION=AGGREGATED_GPU_RUNS
describes the inputs; analysis_process.gpu_inference_performed=false describes the
analysis process. Each condition retains session results, arm statistics and:

- No-Reuse ON minus Replay; RAW ON minus Replay (positive favors Replay).
- No-Reuse ON minus Full; RAW ON minus Full; RAW Replay minus Full (positive
  means that SemCache arm was slower than Full).
- Actual coverage sensitivity between measured adjacent coverage regimes at fixed L.

Session means have equal weights, independent of pair counts. Between-session
variance, wins/rates and seeded session-mean bootstrap 95% intervals are exported.
Only three sessions makes those intervals exploratory. Raw outliers are retained;
negative differences are not clamped. No interpolation, GPU-kernel subtraction,
production/E2E speedup or correctness-qualified service claim is made.

## Local verification

```bash
PYTHONPATH=src:tests python3 -m unittest test_r2c3_impact_ablation test_r2c3_analysis \
  test_r2c1_workloads test_r2c1_benchmark test_r2c1_analysis test_r2c2_profile_memory -q
python3 scripts/92_run_r2c3_impact_ablation.py --help
git diff --check
```

Scalar/mock tests exercise real policy classes, dispatch bypass and restoration,
schema failure paths, order, independent-session analysis and CLI. Dependency-gated
tests cover actual CPU reducer and tiny OPT/PEFT Engine ON/Replay parity. These do
not substitute for OPT-2.7B FP16 CUDA validation. No GPU job is executed locally;
GPU_VALIDATION=NOT_RUN until actual SERAPH inference. No performance finding or
Attention Impact wall-time saving is claimed from tests.

Local verification on 2026-10-28: 21 focused tests, 19 passed and 2 dependency
skips. Combined R1/impact, R2-A, B3.5, C1/C2/C3 regressions: 239 tests, 182 passed,
57 skipped, zero failures. Python compilation, CLI help, Pilot/three latency/
Analyze argument parsing and tracked/new-file whitespace checks passed. Local
Torch/Transformers/PEFT are absent. No GPU experiment, Slurm submission, shell-file
edit, or Git push was performed.
