# R1-B2 RAW reuse latency breakdown and break-even

R1-B2 asks where request time is spent and whether **measured warm-cache** QKV
reuse beats native FULL. These are separate passes: `minimal` is the primary
performance comparison; `breakdown` diagnoses costs. This is an experimental
harness, not an executor or policy optimization and not a safety claim.

Only these files are added:

- `scripts/72_run_r1b2_latency.py`: offline-by-default model execution, isolated
  output, source/target separation, alternating execution order and checkpoints.
- `scripts/73_aggregate_r1b2.py`: compatible-identity summaries and rankings.
- `src/semcache/metrics/r1b2.py`: diagnostic coverage filter, scalar timing schema,
  correctness gate, reporting, checkpoint integrity and aggregation.
- `tests/test_r1b2.py`: dependency-free tests and optional tiny OPT execution tests.
- This document.

Engine, original mixed executor, reducers, Token Precompute, admission/eviction,
CHU/PBR, R1-A/B1/D1 runners, results and `jobs/` are unchanged. No job script is
created. No commit, branch switch or push is part of this work.

## Reused paths and controlled differences

Model/tokenizer loading, controlled LoRA users, tokenizer-aware hotel workload,
TinyBERT encoding, cluster initialization, raw tensor ownership audit, memory
preflight, `raw_only_guard`, native/engine requests, projection row audit and
B1 policy snapshots are reused. Source is the existing **detailed
LOOKUP_NO_REUSE** request, matching B1 source construction. All targets use
`user_a`, the exact source prompt and adapter, eager attention,
`output_attentions=True`, `use_cache=False`, batch one, prefill only.

Token Precompute with `paper_row_l2_sum` and its FP64 reduction are unchanged.
The B1 defaults remain rho .8, history lambda 100, buffered cluster updates every
100 queries, PBR interval 100. Each target has a newly reconstructed source cache;
this two-query fixture does not exercise interval-100 PBR. Resident candidate
identity, policy/history snapshots and selected spans are audited after timing.
This does not simulate a long-running serving cache.

The local repository has no original R1-B1 measurement artifacts. The user's
76.84/74.91 ms FULL and other historical latencies, and D1's FP16 mismatch
observations, are context only, not newly verified numbers. The manifest always
labels historical reference reproduction **UNVERIFIED**. Pin the original
model/tokenizer/encoder revisions using the CLI when comparing with B1, and
compare runtime/code hashes. D1 findings motivate the separate identity and
allclose gates; D1 tensor hooks/full-shape controls are not installed for timing.

## Modes and coverage

|Mode|Path|Coverage control|
|---|---|---|
|FULL_RECOMPUTE|Native OPT/PEFT prefill; no engine/cache|None|
|LOOKUP_NO_REUSE|Existing frontend, empty-hit mixed execution, impact and policy|No physical reuse|
|NATURAL_RAW_REUSE|Original exact-w3-within-cluster selection and mixed execution|Unfiltered; displaced positions remain visible|
|POSITION_ALIGNED_REUSE|Existing strict same prompt/user/adapter/position selector|Runner-only deterministic filter|

`--coverages 0,.25,.5,.75,max` means fractions of **all requested tokens**, not of
available hits. Strict candidates first pass the original selector; the new
filter keeps earliest nonoverlapping windows, at most `floor(length*fraction/3)`.
`max` keeps all feasible strict candidates, not necessarily 100%. Repeated keys
retain their originally admitted source positions; they are not replaced or
force-admitted to manufacture higher coverage. Subsets are nested and
reproducible. Both CPU and GPU storage rebuild the same source and audit its
signature. Candidates are not invented if admission or deduplication limits them.

The filter is a scoped, single-threaded patch of the engine module's **diagnostic
selector binding only**, restored even on exception. Natural selection and
policy implementations are untouched. Filtering changes which hits execute and
therefore can legitimately change subsequent CHU/policy state relative to Natural;
this is an opt-in mechanistic diagnostic, not production-equivalent policy.
`coverage_filter_host_ms` includes filter validation, sorting and mask building;
it is nested in hit selection and request, never added again.

Rows distinguish requested coverage (`null` for unswept modes), actual coverage,
raw hit windows, selected windows, reused/recomputed tokens and actual executed
native/skipped rows **summed across all layers and all three Q/K/V roles**.
Position mismatch count counts tokens; mismatch windows is recorded separately.
FULL minimal has no projection hooks, so its native row count is explicitly
`native_structure_derived_no_inner_hooks`. Engine rows must pass the existing
physical audit. At zero coverage every role/layer must skip zero rows.

## Timing contract

CPU wall uses `perf_counter_ns`; GPU timing uses CUDA stream events. Both request
paths synchronize before starting and before closing the request timer.
Correctness, CPU logits transfers, snapshot/hash packaging and artifact I/O are
outside timing. Model/adapter and encoder initialization are separate manifest
fields. Source build is never added to target warm request latency.

`minimal` uses the unchanged production mixed executor without inner CUDA events.
Existing engine CPU timers, audit records and event logging are still present;
this is **not zero instrumentation**. Engine minimal prefill timing is null:
its asynchronous wall time would not measure GPU completion. Native minimal
prefill wall is available because that helper synchronizes its end.

`breakdown` reuses `r1_runtime.detailed_mixed`, which follows the existing
index/gather/native/scatter algorithm with nested timers. It is a diagnostic
executor, not a new production optimization. All nested events resolve after
request completion; no new per-window synchronization is added. The existing
engine additionally synchronizes at prefill end in breakdown. Blocking CPU cache
copies and impact materialization can themselves synchronize; this behavior is
not removed or hidden. Consequently breakdown-minus-minimal includes Python,
event, detailed-executor and prefill-sync overhead, not just profiler cost.

|Metric family|Clock and interpretation|
|---|---|
|Tokenization, semantic encode, cluster assign/update, subsequence extract, lookup, hit selection|Existing CPU wall intervals; semantic encoding may include GPU waits|
|Cache fetch|Host + CUDA stream intervals around raw tensor `.to(output)`; GPU resident may be a no-op|
|Index transfer|Host + CUDA stream interval for fresh indices|
|Tensor assembly|Host + CUDA stream gather/allocation/scatter/cached slice assignment|
|Native QKV|Host + CUDA stream time for native PEFT projections on fresh rows (all rows for FULL)|
|Mixed executor|Inclusive parent of fetch, index, assembly and native QKV; never add children to it|
|Remaining transformer|**Derived** prefill GPU minus disjoint mixed executor GPU (native QKV GPU for FULL); includes launch/stream idle gaps, not pure kernel compute|
|Attention impact|Existing CPU wall for Token Precompute, host materialization, averaging and history append|
|Cache policy|Existing policy interval includes admission/eviction and materialization; CHU and PBR reported as separate CPU intervals|
|Request wall|Inclusive end-to-end warm request; authoritative latency|

The reused cache-fetch timer measures `.to(output)`, not Python dictionary
pointer retrieval before that call. Index transfer measures `.to(device)`; CPU
mask/fresh-index construction and executor context setup remain within prefill,
not a separately isolated component. Unassigned logging/dispatch/setup costs
are not silently redistributed among measured components.

CPU and CUDA clocks overlap and are **never summed**. Materialization is nested
in cache policy. Derived remaining-transformer time is not added to prefill.
A negative residual is retained as `INVALID_NEGATIVE_DERIVED`, not clamped.
Unavailable/not-applicable metrics are null, not fabricated zeros. A measured
zero-hit mixed request can legitimately have zero cache-fetch work.
The breakdown table is a set of scoped observations, not an additive pie chart.

Minimal/breakdown and CPU/GPU combinations within a repetition compare source
identity, source/target policy snapshots (B1 numerical tolerance and exact
identity/decisions), selected spans, projection rows and logits (`1e-3/1e-3`).
Failures remain in rows and set the run status to `PARITY_FAILED`.
Independent-process aggregation can prove exact equality from hashes; differing
hashes yield `UNVERIFIED_HASH_DIFFERENCE`, not a fabricated tolerance pass/fail.
For tolerant comparison, run `--instrumentation both` with both storage devices
in one process. There is no disk tensor dump or layer capture.

## Correctness and performance gates

The entire output is compared to an untimed native FULL with fixed
`atol=1e-3, rtol=1e-3` (no tolerance CLI). Nonfinite logits cannot pass. Save max
absolute and relative L2 error, allclose, last-token argmax agreement and
KL(FULL || candidate) independently. Source context identity means the entire
source query token sequence, not just the matching window; source position is
checked separately. Natural displaced hits fail the strict gate even if a
particular prediction happens to agree. Strict identity alone does not pass:
FP16 subset GEMM errors remain visible.

- `STRICT_CORRECTNESS_PASS`: nonzero physical reuse, full-logit allclose, matched
  positions, adapter and source context.
- `STRICT_CORRECTNESS_FAIL`: nonzero reuse violating any of those conditions.
- `NO_REUSE`: zero physical reuse; allclose is still recorded separately.
- `UNVERIFIED`: insufficient correctness evidence.

Summary statistics are mean/p50/p95/population std. Each repeat has a FULL of
identical length, tokens, adapter, hardware and instrumentation; paired delta is
reuse minus FULL. `speedup` is paired FULL mean / reuse mean. Modes rotate and
reverse across repetitions; storage/pass order reverses each repetition. This
reduces order bias but does not measure or control GPU clocks/thermal state.

Only minimal nonzero-reuse groups can register observed break-even (ratio >1).
Correctness-qualified break-even additionally requires every paired reused row
to pass strict correctness, every paired FULL to pass its reference check, and
no known within-process comparison-parity failure. Breakdown ratios remain
diagnostic; LOOKUP and zero-coverage cases never count as reuse break-even.
No significance or long-context extrapolation is implied by a ratio above one.
If none appears, report `NOT_OBSERVED_IN_TESTED_RANGE`, with actual measured
lengths; failures and unmeasured lengths are not silently treated as passes.

Save `source_build_ms` and `request_wall_ms` separately. For a hypothetical N
uses of an unchanged source, the explicitly modeled per-target cost is
`source_build_ms/N + warm_request_ms`, and total is
`source_build_ms + N*warm_request_ms`. This is amortization arithmetic, not a
measurement of N sequential requests or unchanged long-run policy state.

## CLI

No download happens by default (`local_files_only=True` for both model/tokenizer
and encoder). `--allow-download` is the explicit opt-in. On SERAPH use the existing
`semcache` environment and local cache:

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

32-token smoke (FP32 correctness control, reduced repetitions):

```bash
python scripts/72_run_r1b2_latency.py --profile smoke \
  --model facebook/opt-125m --dtype float32 --prompt-lengths 32 \
  --cache-devices cpu,cuda --instrumentation both \
  --warmup-runs 1 --measured-runs 2 \
  --output-dir results/r1b2/smoke125m_v1
```

CPU model smoke, if Torch/Transformers/PEFT and all model/encoder weights exist:
add `--device cpu --cache-devices cpu`. Missing packages/weights fail explicitly;
no fake model benchmark substitutes for them.

32/64-token pilot, GPU resident first:

```bash
python scripts/72_run_r1b2_latency.py --profile pilot \
  --model facebook/opt-2.7b --dtype float16 --prompt-lengths 32,64 \
  --cache-devices cuda --instrumentation both \
  --output-dir results/r1b2/pilot27b_v1
```

Formal run (defaults warmup 3, measured 10, all five coverage settings):

```bash
python scripts/72_run_r1b2_latency.py --profile full \
  --model facebook/opt-2.7b --dtype float16 \
  --prompt-lengths 32,64,128,256 --cache-devices cpu,cuda \
  --instrumentation both --coverages 0,.25,.5,.75,max \
  --output-dir results/r1b2/full27b_v1
```

Each default storage/pass/repeat group has 8 targets, 7 fresh source builds and
one untimed native reference. Four lengths and four storage/pass combinations
with 13 repeats therefore mean **1664 targets + 1456 sources + 208 references**.
Use smoke/pilot first. Long runs should be launched through the user's own Slurm
setup; this change provides no Slurm files or walltime estimate.

Length and instrumentation can be split without pooling incompatible identities:

```bash
python scripts/72_run_r1b2_latency.py --profile full --prompt-lengths 256 \
  --cache-devices cpu,cuda --instrumentation minimal \
  --output-dir results/r1b2/length256_minimal_v1
python scripts/72_run_r1b2_latency.py --profile full --prompt-lengths 256 \
  --cache-devices cpu,cuda --instrumentation breakdown \
  --output-dir results/r1b2/length256_breakdown_v1
python scripts/73_aggregate_r1b2.py \
  --inputs results/r1b2/length256_minimal_v1 results/r1b2/length256_breakdown_v1 \
  --output-dir results/r1b2/aggregate256_v1
```

For all lengths, aggregate the formal root instead with
`--inputs results/r1b2/full27b_v1`. Supply several independent roots as needed.
Smoke and formal results should be reviewed separately; sample counts are always
shown. Identical paired rows/overlapping inputs are rejected, not counted twice.
Different code/model/adapter/config/hardware identities remain separate groups.

To resume, repeat the **exact original command** with `--resume`. It verifies
code/config/runtime identity, reloads checksummed atomic pairing groups and
rewarms before unfinished measured groups. Incomplete groups rerun entirely;
no partially measured group is committed. A completed rerun without `--resume`
fails rather than overwriting. Changed code/config requires a new output root;
never reuse R1-A/B1/D1 directories. Aggregation also requires a new directory.

## Artifacts, failure states, and memory

Every `results/r1b2/<tag>/tokens_<length>/` contains:

- `manifest.json`: code SHA256/HEAD/branch/worktree, resolved model/tokenizer/
  encoder provenance, controlled adapter metadata, seed, GPU UUID/driver,
  Torch/CUDA/backend flags, initialization times, settings, raw-only import and
  tensor evidence, preflight, status and attempts.
- `raw.jsonl`: scalar request/source times, coverage, identity/signatures,
  compact span hashes, memory snapshots, correctness and timing scopes.
- `checkpoints/*.json`: atomic complete pairing groups with checksum.
- `latency_breakdown.csv`, `break_even_summary.csv`, `break_even_status.json`.
- `correctness.json`, `coverage_audit.json`, `instrumentation_overhead.json`.

Aggregate adds `overall_summary.csv`, `break_even_matrix.csv`,
`bottleneck_ranking.csv` and an input-status manifest. Ranking separates CPU
top-level costs from GPU subregions; it excludes inclusive mixed/pre-fill GPU
parents from child ranking. It cannot discover a bottleneck without measurements.
Raw/checkpoints duplicate only compact records; no attention/hidden/QKV/logit
tensors are written. Temporary CPU logits are retained for pair checks only.

Existing conservative memory preflight includes eager attention/reduction
workspace, activations/logits, temporary GPU cache overflow and reserve; it is
not an OOM guarantee. Failed screening is `MEMORY_PREFLIGHT_FAILED`; allocator
OOM is `OOM`. Requested 256 is never silently reduced. Exact tokenization
mismatch is `TOKEN_LENGTH_MISMATCH`. Interruptions preserve completed groups and
write `INTERRUPTED` when catchable; hard termination may leave `RUNNING`, which
aggregation rejects by default. `--allow-incomplete` explicitly enables partial
aggregation, preserving `PARTIAL_INPUTS` and failed input statuses. It does not
make a missing length part of the tested range. Output collisions fail.

Lengths >=512 are accepted as a future explicit setting, not part of this formal
study; they still require memory preflight and model positional support. No 4K,
8K,16K/Qwen/QMSum claim follows from these utilities.

## Tests and current evidence

```bash
PYTHONPATH=src:tests python3 -m unittest \
  test_r1b2 test_r1b1_d1 test_r1b1_impact test_m9a2_impact_audit \
  test_m85_alignment test_r1_raw_semcache -v
python3 -m py_compile scripts/72_run_r1b2_latency.py \
  scripts/73_aggregate_r1b2.py src/semcache/metrics/r1b2.py tests/test_r1b2.py
```

At implementation time the local environment has no Torch/Transformers/PEFT.
The suite passed 50 tests with 22 explicit dependency/GPU skips (72 total);
R1-B2 contributes 12 dependency-free passes and two model skips. Failure messages
inside lifecycle tests are deliberately injected test cases, not GPU runs.
Optional tiny OPT tests check actual row skipping and minimal/breakdown policy,
residency-selection and logit equivalence on CPU FP32 and CUDA FP16/CPU-GPU cache.
They have **not run here**. Additional M8 metrics and mixed-semcache test
collection failed because `pytest` is absent; those suites are not reported as
passed. On SERAPH run `PYTHONPATH=src:tests python -m pytest
tests/test_m8_metrics.py tests/test_mixed_semcache.py` (one shell line).
An actual no-Torch CLI invocation returned exit 1 and preserved the `FAILED`
manifest and empty reports in a temporary directory. No OPT-125M or OPT-2.7B benchmark or actual break-even
was measured locally. Historical D1 numerical values are not newly reproduced.

R1-B3 candidates remain hypotheses pending minimal/breakdown data: per-window
CPU work, raw transfer/index/assembly costs, mixed projection launch overhead,
semantic frontend, attention impact, and cache materialization/policy. Select a
candidate from measured rankings and validate correctness separately; this
implementation does not optimize any of them.
