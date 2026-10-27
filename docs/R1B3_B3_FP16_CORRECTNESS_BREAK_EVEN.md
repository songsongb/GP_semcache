# R1-B3-B3: FP16 divergence and FULL-relative break-even

Implementation status: experiment tooling ready; real model/GPU validation is not
performed locally (Torch/Transformers/PEFT are absent). User-provided B2.1 numbers
are historical inputs, not reproduced B3 measurements. No FP16 cause or speedup is
claimed before these runners execute. Classification remains `ORACLE_DIAGNOSTIC`.

## Existing execution and identity

The checked local branch is `exp/raw-semcache-r1b3`; manifests record actual Git
HEAD/branch/status plus SHA256 for Python sources/scripts, rather than assuming a
branch name. Existing executors, B2.1 runner, policies and historical results are
unchanged. Dynamic B2.1 helpers supply the validated source-engine clone and
`RequestMapping`. D1 supplies FP64 error diagnostics, same-hidden Full/Subset
shape probes, and the optional full-shape injection correctness control.

Native FULL uses `native_request`: same OPT/controlled PEFT model, actual tokenizer,
IDs, eager attention, `eval()`, `output_attentions=True`, `use_cache=False`, batch 1.
Reuse uses `engine_request`, the B1 Fast Selector, source-only occurrence catalog,
and existing Legacy/V1/V2/Contiguous executor contexts. Source and target both use
`user_a`; `--different-adapter-control` separately runs `user_b` FULL and never
injects its QKV into `user_a`. Namespace, resolved model/tokenizer revision, both
adapter weight hashes, dtype/device, dropout/eval, TF32/determinism settings,
source snapshot and actual target tokens are recorded.

Source cache is built exclusively from passive actual source-prefill TOTAL QKV.
Target FULL results are references only. CacheGen and quantized models are rejected.
The existing snapshot/backing/adapter/prefix guards and projection-row audit run
for every reuse request. Strict tolerance stays `atol=rtol=1e-3`.

## Correctness runner: script 79

The default comparisons are FULL_A/FULL_B, SOURCE_FULL/TARGET_FULL,
TARGET_FULL/LOOKUP_NO_REUSE, TARGET_FULL/Legacy, TARGET_FULL/Contiguous and
Legacy/Contiguous. Optional V1/V2 are compared with both Full logits and Legacy
QKV/logits. LOOKUP uses the actual Engine `SEMCACHE_LOOKUP_NO_REUSE` path.

Hooks observe token and position embeddings, layer input/output, attention output,
QKV input, actual base projection output, scaled actual LoRA B output, and TOTAL
QKV output. They install **after** executor context validation, remove in `finally`,
and never recompute inside the inference path. Base/LoRA candidate rows are aligned
with the actual fresh absolute positions; skipped components of fully reused rows
are not fabricated as zero-cost/zero-error observations.

One CPU Target-Full reference is retained. Candidate hook outputs are immediately
compared and released. Legacy adds one temporary CPU reference for direct executor
parity. No per-layer GPU history is retained; no tensors are written to disk.
`--snapshot-limit-mib` limits each host reference (default 1024 MiB, at most two
simultaneous references). Selected `--layers` scope the claim to the first
**observed** divergence in those layers; `all` covers the full model.

Outputs report max/mean absolute error, relative L2, first differing and first
failing token/coordinate, element-wise tolerance violations/count/fraction,
argmax agreement, last-position KL, top-k reference logits and the top1/top2
margin relative to twice the last-position maximum error. Torch's existing
allclose operand order is retained: `allclose(reference, candidate)`.

The root report keeps H1–H5 distinct:

- H1: D1 reruns Full and Subset GEMMs on identical Target-Full hidden inputs,
  separately for total PEFT, base, LoRA A and B with identical A input.
- H2: actual source hook QKV is checked exactly against catalog backing, then
  source prefill QKV is compared with native Target-Full QKV.
- H3: same adapter and causal-prefix provenance are enforced; optional different
  adapter Full results are labeled separately.
- H4: per-layer QKV/attention/hidden-error trajectory documents propagation.
  That trajectory alone does not isolate a unique numerical cause.
- H5: repeated native Full outputs and hook values are compared. Two exact Full
  runs mean nondeterminism was not observed in that control, not proven absent.

`--full-shape-control` reuses D1 native-full projection plus source overwrite.
It skips **zero** projection rows and is only a correctness control. Its hooked
runtime is not a serving or optimized-reuse latency measurement. No numerical
remediation policy or tolerance relaxation is introduced.

Required artifacts: `manifest.json`, `baseline_repeatability.json`,
`first_divergence.csv`, `layerwise_qkv_comparison.csv`, `precision_control.csv`,
`logit_correctness.csv`, `root_cause_summary.json`. A run may finish with
`execution_status=COMPLETE` and `correctness_status=STRICT_FAIL`. OOM, missing
capture evidence, executor parity failure and runtime exceptions have distinct
execution failure statuses and complete traceback/context.

## Minimal benchmark: script 80

Primary comparison is FULL vs Contiguous. Legacy is also paired to verify exact
executor output equality. `--executors all` adds V1 and V2. FULL executes inside
every paired repeat, with the same actual target IDs. Warmup and measured order
schedules are independently balanced seeded Latin blocks. `--seed` fixes inputs,
model and adapters; distinct `--order-seed` values vary independent-run execution
order. The complete schedule is stored. No outliers are removed.

`--profile smoke` defaults to 1/2 warmup/measured; `pilot` to 3/10; `formal` requires
at least 3/30. All counts are explicit in manifests. Request wall milliseconds are
synchronized by existing R1 functions. FULL prefill wall is also retained. Minimal
has no extra numerical hook, profiler or per-window synchronization. Existing
passive QKV outputs are copied for exact validation **after the whole timing
block**, for every distinct target and executor. Copies never enter performance
samples or build the source cache. Row audit and paired logits are checked for
every timed pair. QKV validation failures invalidate all affected samples.

`--instrumentation breakdown` is a separate run/directory. It uses B2 ExecutorBook
and B2.1 SetupBook. Nested host/CUDA intervals remain diagnostic, are never added
to request wall time, and cannot generate a break-even success claim. Cold index
CUDA events are resolved only after enclosing request synchronization. Compare
minimal and breakdown directories descriptively to inspect instrumentation cost;
the independent-run aggregator rejects mixing their configurations.

### Target workloads

- `warm`: original text/IDs repeated; first mapping miss occurs in warmup and
  measured identical requests can reuse exact-target metadata.
- `cold`: change only the final, unreused token. Decode/re-encode must reproduce
  every exact ID and preserve length. Source-cluster preflight and actual timed
  Fast Selection must retain feasible reuse. Each token sequence is new; a mere
  request-ID change never counts as cold. A positive-reuse V2/Contiguous mapping
  hit in this workload is rejected.
- `mixed`: alternate original target and previously unseen valid suffix changes;
  measured actual hits/misses, not an assumed 50% rate, define the result.
- `noncontiguous`: runner-only opt-in filter retains every second already-selected
  nonoverlapping window. It invents no candidates, preserves prefix/identity and
  triggers existing gather fallback. Actual coverage is lower and explicitly
  reported. This is not the unfiltered production selection policy.

The cold fixture requires `length % 3 != 0`, so an unused final token exists even
at MAX (128 and 256 satisfy this). Fixture generation is offline workload creation;
actual tokenizer, encoder, cluster and selection still run inside each request.
No pretokenized proxy or forced cluster is used. Failure to construct enough valid
fixtures is `MISSING_EVIDENCE`, never silent reduction of measured count.

Each length/coverage/workload case has a source snapshot shared by all compared
executors, with separate exact-target metadata caches per executor. Cases can run
in separate invocations. Existing outputs are refused; this new runner deliberately
has no resume/overwrite option. Scalar checksummed per-case checkpoints preserve
completed timing evidence after interruption; incomplete runs are rejected by
aggregation. Source build cost, physical bytes, index/descriptor estimates, target
mapping statistics and temporary/allocator memory remain separately recorded.

### FULL-relative costs and gates

`delta = FULL_minimal - REUSE_minimal`; positive means reuse is faster. Summaries
retain mean, median, p95, standard deviation, min/max, win count/rate and actual
coverage. Observed speedup and strict correctness qualification are independent.
Strict FAIL can remain a completed diagnostic, but cannot qualify break-even.
Zero-reuse rows cannot qualify as a correct physical-reuse speedup.

Amortization solves the strict inequality

```
N * FULL > S_incremental + REUSE_cold + (N - 1) * REUSE_warm
```

No finite steady-state break-even is reported when `FULL <= REUSE_warm`.
`S_incremental` is measured Catalog + FastIndex + (V2/Contiguous) Descriptor build.
Catalog copy is included once in Catalog build, not added twice. Separate fields
report source query cost and the included source-side storage copy.

The existing-source scenario assumes **prefill and capture are already sunk**.
It is conditional: extra capture/frontend costs of converting an ordinary native
source into a reuse-ready source are not isolated by the existing detailed source
runner. The dedicated-source scenario adds that entire measured source query cost.
Source timing uses the existing detailed LOOKUP source build and is labeled as
such; it is not a minimal native-source baseline. Costs are assigned to one source
snapshot shared by N targets. Cold samples observed during warmup are reported
with sample counts; a single cold sample does not establish a robust cold-latency
mean. These N estimates are derived cost models, not observed long-horizon runs.
Mixed/unseen-target workloads do not inherit identical-target amortization claims.
B2.1's Legacy-relative 1–2-request estimate is not used here.

GPU state is sampled at block boundaries outside requests. The monitor resolves
the actual Torch device UUID before querying utilization, clocks, temperature,
power, memory and process listing. Unsupported monitoring stays null with a
reason; it never silently labels physical GPU 0 as a different CUDA device.

Artifacts: manifest, latency/paired/coverage/cold-warm/source-cost/memory CSVs,
`gpu_state.json`, `break_even_summary.json`, `summary.json`, exact validation
status in `correctness_comparison.csv`, optional `executor_breakdown.csv`, and
small per-case checkpoints. No attention, hidden or QKV dumps.

## Independent runs: script 81

The aggregator requires complete manifests, matching code/model/tokenizer/adapter/
workload/hardware/instrumentation identities, every case checkpoint and every
paired repeat. Run IDs **and order seeds must differ**. It resamples independent
runs, then paired repeats within each run, weighting runs equally. Outputs include
hierarchical bootstrap 95% CI, between-run standard deviation, raw differences,
statistics and correctness gates. A reproducible claim requires >=3 runs,
>=30 pairs/run, verified GPU UUID, minimal instrumentation, exact executor parity
and CI lower bound >0. Correctness qualification additionally requires strict
PASS and actual positive reuse. Scalar synthetic test fixtures are not benchmarks.

## SERAPH execution

Use the user's existing scheduler for long runs. This change adds no Slurm files.
Configure the existing local Hugging Face cache:

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

No model downloads occur unless `--allow-download` is explicitly passed.
Start with OPT-125M FP32 correctness and small benchmark smoke:

```bash
python scripts/79_diagnose_r1b3_b3_fp16.py \
  --model facebook/opt-125m --dtype float32 --prompt-lengths 32 \
  --full-shape-control --different-adapter-control --secondary-executors \
  --output-dir results/r1b3/b3/correctness/smoke125m_v1

python scripts/80_benchmark_r1b3_b3_break_even.py \
  --profile smoke --model facebook/opt-125m --dtype float32 \
  --prompt-lengths 32 --coverages 0,.5,max \
  --workloads warm,cold,mixed,noncontiguous --executors all \
  --order-seed 101 --output-dir results/r1b3/b3/break_even/smoke125m_v1
```

Then diagnose OPT-2.7B FP16 and pilot MAX only:

```bash
python scripts/79_diagnose_r1b3_b3_fp16.py \
  --model facebook/opt-2.7b --dtype float16 --prompt-lengths 128,256 \
  --full-shape-control --different-adapter-control \
  --output-dir results/r1b3/b3/correctness/opt27b_fp16_v1

python scripts/80_benchmark_r1b3_b3_break_even.py \
  --profile pilot --model facebook/opt-2.7b --dtype float16 \
  --prompt-lengths 128,256 --coverages max --workloads warm \
  --executors primary --instrumentation minimal --order-seed 101 \
  --output-dir results/r1b3/b3/break_even/pilot_v1
```

Optional FP32 control after preflight, initially 32 tokens only:

```bash
python scripts/79_diagnose_r1b3_b3_fp16.py \
  --model facebook/opt-2.7b --dtype float32 --prompt-lengths 32 \
  --snapshot-limit-mib 2048 --full-shape-control \
  --output-dir results/r1b3/b3/correctness/opt27b_fp32_v1
```

Raw parameter storage is approximately 5.4 GB (FP16) or 10.8 GB (FP32) before
activations, attentions, LoRA, encoder, catalog and allocator reservation. These
are parameter-count estimates, not observed peaks. A pre-load screen reserves
3 GiB beyond that estimate; post-load R1 memory preflight runs per length. For
2.7B all-layer CPU captures, provide >=2 GiB host headroom at FP16, more for FP32,
in addition to model loading; the configurable snapshot budget can fail early.
Actual OOM remains a recorded failure; 24 GB does not guarantee a configuration.

After pilot validation, formal coverage/workload sweep (split lengths or workloads
into separate jobs/output roots when needed):

```bash
python scripts/80_benchmark_r1b3_b3_break_even.py \
  --profile formal --model facebook/opt-2.7b --dtype float16 \
  --prompt-lengths 128,256 --coverages 0,.25,.5,.75,max \
  --workloads warm,cold,mixed,noncontiguous --executors all \
  --instrumentation minimal --warmup-runs 3 --measured-runs 30 \
  --seed 42 --independent-run-id run01 --order-seed 101 \
  --output-dir results/r1b3/b3/break_even/formal_run01
```

Repeat identical settings with `run02 / --order-seed 202` and
`run03 / --order-seed 303`, using new `formal_run02` and `formal_run03` directories.
Use separate `--instrumentation breakdown` outputs for attribution only.

```bash
python scripts/81_aggregate_r1b3_b3.py \
  --run-dirs results/r1b3/b3/break_even/formal_run01 \
             results/r1b3/b3/break_even/formal_run02 \
             results/r1b3/b3/break_even/formal_run03 \
  --output-dir results/r1b3/b3/aggregate/formal_v1 --seed 9107
```

## Local checks

```bash
PYTHONPATH=src:tests python3 -m unittest \
  test_r1b3_b3_correctness test_r1b3_b3_break_even -v
```

Pure CPU tests cover first-divergence event ordering, fixed tolerance metrics,
FULL-relative strict-inequality cost arithmetic, cold token identity/prefix,
balanced schedules, bootstrap, output collision, incomplete evidence, strict-fail
gates and independent-run identities. Optional Torch tests exercise actual tensor
metrics and base/LoRA/downstream hooks. Missing Torch/CUDA tests are SKIP; CLI
startup failure artifacts are not completed model smokes.

Local verification for this implementation: 136 tests collected, 93 passed and
43 skipped, no failures (existing R1/B1/D1/B2/B3-A/B3-B1/B2/B2.1/M9-A.2 regression
coverage included). The B3-only suite collected 13 tests: 11 passed, 2 Torch tests
skipped. Python compilation and all three CLI help commands pass. Both real
smoke CLI startup attempts terminate at missing Torch with `execution_status=FAILED`
and `correctness_status=UNVERIFIED`, preserving the required artifact schemas and
traceback. No OPT/GPU numerical result, memory peak, latency or break-even has
been measured here. CPU quality calculations occur outside request clocks between
paired repeats; these pauses may affect GPU clocks, so monitor state and use the
independent-run study rather than interpreting a single short smoke as performance.
