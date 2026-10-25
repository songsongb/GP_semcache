# R1-B1: M9-A.2 attention impact integration

R1-B1 integrates the existing M9-A.2 FP64 arithmetic into the real Raw SemCache
engine. It is an opt-in execution algorithm, not a new reducer formula. There
are no local GPU measurements, no claimed latency/speedup numbers, and no safety
claim. CacheGen, quantization, Qwen, and mixed-QKV optimization are excluded.

## Reused implementation and engine connection

The arithmetic formerly inside `diagnostics.impact.measure_impact` is now in
`cache.impact_execution.compute_window_values`. The extracted `values_out`
assignment and calculation branches are AST-identical to their prior version;
the production `PaperRowL2SumReducer.reduce` is untouched.

* `CURRENT_BLOCKWISE` retains M9's per-window/layer reduction and counters.
* `TOKEN_PRECOMPUTE` retains the per-token FP64 layer sum, candidate membership
  mask, window reduction and one packed finite-flag/value host materialization.
* `TOKEN_PREFIX_SUM` retains the deterministic doubling scan and indexed prefix
  differences. This optional path is supported, but must pass the same gates.

The common helper has no history, query ID, cache policy, or finalization. M9's
diagnostic wrapper still calls its own `finalize`, including its historical
`impact_audit` identity, on its diagnostic history. Engine never calls that
wrapper or finalizer. The extraction adds shared-function dispatch/validation
overhead to new standalone M9 timing runs; historical timing files are not
rewritten or claimed bitwise timing-equivalent. Arithmetic, result ordering,
finite handling, diagnostic operation counters, and finalization semantics
are preserved.

Engine accepts independent constructor options:

```python
SemCacheEngine(...,
    impact_reducer="paper_row_l2_sum",       # mathematical formula
    impact_execution="token_precompute",   # execution algorithm
    capture_impact_values=True)             # optional outside-timer audit output
```

`impact_execution` defaults to `legacy_blockwise`, which still calls the
original reducer once per candidate, including the original per-window CPU
mask construction. Existing M7/M8/M9 callers retain this default. Optimized
execution requires the exact, unmodified `PaperRowL2SumReducer` class; custom
subclasses and other reducers are rejected instead of silently bypassed.

The optimized branch receives actual `output.attentions`, all candidate windows
in their original order, and the same validity mask. FP64 promotion, causal and
padding interpretation, masked/nonselected nonfinite handling and input tensor
immutability remain M9 semantics. Engine performs its original per-key mean for
duplicate token subsequences, then its original **single**
`history.append(cluster, query_id, cache.now, per_query)` call. CHU, admission,
eviction and scheduled PBR remain after this common path. No model forward is
added to compute impact. Values are packaged for audit after request timing
closes, without a second reduction or extra tensor transfer.

## Benchmark integration and boundaries

`scripts/70_run_r1b1_impact.py` uses a small extension interface in the existing
R1-A runner rather than copying model setup, cache reconstruction or timing.
Without an extension, script 69 keeps its CLI defaults and execution schedule.
Neither `r1_runtime.py` nor either mixed executor is modified in this stage.

Default B1 algorithms are `legacy_blockwise,token_precompute`. Add
`,token_prefix_sum` to `--impact-executions` to include the prefix diagnostic.
Every run requires legacy plus at least one optimized algorithm. Algorithms
run in the same loaded-model process, in adjacent per-storage/pass blocks,
with order reversed on alternating repeats. Mode order rotates as in R1-A.
These are paired reconstructed-state measurements, not simultaneous requests;
thermal, allocator and block-order effects remain possible.

For each case/mode/algorithm, source construction uses a new engine and that
algorithm's source impact path. Source snapshots must match legacy identities
and policy decisions, with numerical scores/impacts within tolerance. Source
build costs and source policy snapshots remain separate from target latency.
Every target compares logits to the corresponding untimed native FULL. FULL
is repeated in each algorithm block; its own execution has no impact reducer.

R1 invariants remain: one model/tokenizer/adapter pair, identical token IDs,
eager attention, `use_cache=False`, `output_attentions=True`, no generation,
raw CPU or GPU resident TOTAL Q/K/V and physical fresh-row-only projections.
The raw-only import guard, unquantized model/projection checks, memory preflight,
peak GPU memory and residency accounting are reused.

Minimal request latency is the primary performance metric. Minimal still
retains R1's original CPU timers and audit/event work. B1 reads the existing
attention-impact CPU timer without adding reduction/event instrumentation.
On CUDA minimal, impact wall time can include queued prefill completion;
breakdown synchronizes prefill first and is the better attribution diagnostic.
The detailed pass remains R1's instrumented mixed executor and measures its
additional overhead. No nested GPU/CPU subintervals are added to request time,
and no instrumentation delta is automatically subtracted.

Each summary group separates storage, pass, requested length, condition, mode,
and algorithm. It reports request/impact mean, p50, p95, population std, matched
algorithm-minus-legacy deltas, matched request-minus-FULL deltas, observed FULL
mean/request mean ratio, physical coverage and skipped rows. `observed_full_request_ratio`
is a timing ratio even for an incorrect natural reuse result, **not validated
acceleration**. `validated_strict_reuse_ratio` is populated only for strict
nonzero reuse cases with full logit and algorithm policy checks passing.
M9's isolated compute speedup is never substituted for end-to-end timing.

## Validation gates and evidence

For source and target, artifacts retain:

* every candidate window's identity, order and computed impact;
* lookup HIT/MISS, selected masks/spans and source/destination provenance;
* admission scores/decisions, insertion, denial, deduplication and eviction;
* CHU and PBR events, bounded impact history with query IDs and clock values;
* ordered resident keys/positions, impacts, frequency/access/update state;
* executed projection records and actual reused token counts.

Numerical impact/score/history comparisons use the M9 rule
`abs(candidate-reference) <= 1e-10 + 1e-12 * abs(reference)`. Boolean decisions,
keys, selected positions, history IDs/clocks, counts and projection rows must
match exactly. The comparator records differences with paths and both values;
a tiny numeric error does not excuse a flipped admission or eviction decision.
Source and target checks are both required. CPU/GPU storage and minimal/breakdown
policy evidence are also compared independently, with failures retained.

FULL logit comparisons retain R1's explicit default `torch.allclose` atol/rtol
1e-3 and report max/relative errors, KL and argmax agreement. Tolerances can be
set explicitly. New tiny-FP32 model tests use atol 1e-6 / rtol 1e-5.
Only `POSITION_ALIGNED_DIAGNOSTIC` + `same_user_exact` with nonzero physical reuse
can count as a validated reuse fixture. Natural same-user repeated-subsequence
and cross-user w3 hits are never counted as safe successes, even if one observed
logit comparison happens to pass. General source/destination positional/context
differences remain in `reuse_provenance` and `subsequence_audit`.

Final `policy_equivalent_optimization=true` requires complete algorithm pairs,
all policy/numeric checks, no strict/FULL/lookup/cold logit failures, and at least
one measured nonzero strict fixture for **each** algorithm. It is scoped to the
executed fixtures; it is not proof for arbitrary contexts. Failed comparisons
return nonzero with `POLICY_PARITY_FAILED`, `CORRECTNESS_FAILED`,
`NO_VALIDATED_REUSE`, or `INCOMPLETE_PAIRING`. All validation failure categories
are retained in the manifest. Execution/OOM failures use `FAILED` and preserve
already-completed checkpoints.

Default source+target traces have only two queries. With PBR interval 100 they
do not exercise scheduled PBR; the manifest states `pbr_exercised=false`.
Likewise default capacity may not evict. New tiny-model tests explicitly use
PBR interval 1, bounded history and a two-block capacity to exercise these
paths. Passing a non-evicting/default trace alone does not certify boundaries.

## Tests

No model download is needed for synthetic tensor or tiny random OPT tests.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_r1b1_impact.py -v
PYTHONPATH=src python3 -m unittest discover -s tests -p test_m9a2_impact_audit.py -v
PYTHONPATH=src python3 -m unittest discover -s tests -p test_m85_alignment.py -v
python3 -m unittest discover -s tests -p test_r1_raw_semcache.py -v
```

On SERAPH additionally run the existing pytest regression suite:

```bash
PYTHONPATH=src python3 -m pytest -q tests/test_m8_metrics.py tests/test_m85_alignment.py \
  tests/test_m9a2_impact_audit.py tests/test_r1_raw_semcache.py tests/test_r1b1_impact.py
```

Tests cover defaults/reducer compatibility, shared-code/history separation,
exact decision rejection near an admission boundary, failure retention,
checkpoint isolation, randomized FP16/FP32/FP64 attentions/windows (including
CUDA when available), candidate ordering, duplicate keys, causal/padding masks,
nonfinite valid/invalid cells, tiny-OPT strict logits, residency, projection rows,
CHU, PBR, bounded history and eviction. The tensor-free engine plumbing test uses
explicit stubs and does **not** claim numerical inference validation.

Local Ubuntu has no torch/transformers/peft/pytest. Therefore tensor, model,
CUDA and pytest tests cannot validate numerical equivalence here. Stdlib
contracts and available M8.5/M9-A.2/R1-A regressions run locally; GPU latency
and optimization success remain unverified until the SERAPH runs finish.

Local validation results:

|Suite|Passed|Skipped|
|---|---:|---:|
|R1-B1|6|5|
|M9-A.2|10|8|
|M8.5|9|2|
|R1-A|9|2|

Python compilation, B1 CLI help, `bash -n` on the sbatch template and
`git diff --check` also passed. Skipped tests require absent tensor/model/CUDA
dependencies. The pytest-only M8 metrics suite is unexecuted because pytest
is absent; its SERAPH regression command is provided above.

## SERAPH smoke

Use the existing `semcache` environment from `/data/khuss/repos/GP_semcache`:

```bash
conda activate semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

srun --partition=batch_eebme_ugrad --gres=gpu:1 --cpus-per-task=4 \
  --mem=48G --time=00:50:00 \
  python3 -u scripts/70_run_r1b1_impact.py \
  --model facebook/opt-125m --dtype float32 --device cuda:0 \
  --prompt-lengths 32 --cache-devices cpu --instrumentation both \
  --impact-executions legacy_blockwise,token_precompute \
  --warmup-runs 1 --measured-runs 2 --smoke \
  --output-dir results/r1b1/opt125m_32_smoke_v1
```

The 50-minute smoke allocation is a cap, not a runtime estimate. Run the tests
with a GPU allocation as well to exercise their CUDA paths. If the smoke is
interrupted, use its same arguments/output directory plus `--resume`.

## Formal OPT-2.7B runs and restart

`jobs/r1b1_impact.sbatch` follows the repository's existing partition and conda
conventions. It requests one GPU, 48 GB host memory and **23:50:00**, with one
array task each for 32 and 64 tokens. Array concurrency is capped at one.
It defaults to FP16, CPU/GPU storage, both instrumentation passes, two impact
algorithms, 3 warmups and 10 measured repeats. No runtime duration is claimed.
Long formal experiments are submitted via sbatch, not launched directly here.

```bash
sbatch --export=ALL,R1B1_RUN_TAG=opt27b_impact_v1 jobs/r1b1_impact.sbatch
```

Outputs are `results/r1b1/opt27b_impact_v1/tokens_32` and `tokens_64`.
For separate CPU-only or minimal-only studies, use a new tag:

```bash
sbatch --export=ALL,R1B1_RUN_TAG=opt27b_cpu_minimal_v1,R1B1_STORAGE=cpu,R1B1_INSTRUMENTATION=minimal \
  jobs/r1b1_impact.sbatch
```

If length 64 reaches the walltime limit, restart only its task, with unchanged
code/config/runtime identity. Completed measured groups are skipped; warmups
rerun and incomplete groups restart from fresh source state:

```bash
sbatch --array=1 --export=ALL,R1B1_RUN_TAG=opt27b_impact_v1,R1B1_RESUME=1 \
  jobs/r1b1_impact.sbatch
```

For prefix-sum testing set `R1B1_EXECUTIONS` in the submitting environment (it
contains commas, so avoid embedding it in Slurm's comma-separated export list):

```bash
export R1B1_EXECUTIONS=legacy_blockwise,token_precompute,token_prefix_sum
sbatch --export=ALL,R1B1_RUN_TAG=opt27b_prefix_v1 jobs/r1b1_impact.sbatch
unset R1B1_EXECUTIONS
```

The preflight is a conservative screen, not an allocation guarantee on the
3090 24 GB. Work is split by length; if needed, split storage/pass using the
environment options and distinct tags. Inputs 128/256 remain available as
explicit follow-up diagnostics for repeated subsequences, not part of the
formal 32/64 plan. 512+ is still rejected. There are no Qwen or dataset changes.

## Files and compatibility

Each new output root contains `manifest.json`, atomic `checkpoints/*.json`,
`raw.jsonl`, `summary.csv`, `summary.md`, `policy_parity.json`,
`storage_instrumentation_parity.json`, `correctness.json`, and
`instrumentation_overhead.json`. Policy failures are reported, not filtered.
Before both algorithms complete a pair, reporting explicitly says PENDING.
The manifest hashes engine/common math/runner code and records model/tokenizer/
encoder revisions, synthetic adapter seeds, driver/GPU, memory and raw-only
evidence. The B1 wrapper and sbatch template are included in code provenance.

Use **new R1-B1 output roots**. R1-A checkpoint identity includes local code
hashes; this integration changes them, so old R1-A `--resume` is expected to be
rejected. No old M8/M9/C9/R1-A files or result directories are rewritten. The
legacy R1-A script still defaults to legacy impact; only explicit B1 execution
selects optimized algorithms.
