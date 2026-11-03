# R5-I5R-2B-S1: isolated native Linear shape control

IMPLEMENTED / CPU_FIXTURES_ONLY / GPU_NOT_RUN. The mechanism remains
ROOT_CAUSE_UNRESOLVED. NUMERICAL_FIDELITY_RESTORED is NOT_EVALUATED.

The researcher's verified job 152846 localized the earliest recorded difference
to Layer 0 native GPU base Linear outputs with matching prefix inputs and CPU
LoRA deltas. Those observations are a handoff, not findings reproduced locally.
The original run and `block_0_152846_arithmetic_review_02` are on SERAPH; this
implementation does not invent their parent paths or require transfer locally.

## Files and reuse

New modules `r5_i5r2bs1_plan.py`, `r5_i5r2bs1_runtime.py` and
`r5_i5r2bs1_analysis.py` provide historical operand verification, execution and
independent comparison. Scripts 133/134/135 and `jobs/r5_i5r2bs1_linear.sh` are
the operator interface. `tests/test_r5_i5r2bs1.py` is synthetic CPU coverage.

No production source file is modified. Existing `load_bare()`,
`OPTModelAdapter.projection_modules(0)`, `validate_prepared()`, complete base
fingerprints, GPU allocation diagnostics, recorded backend observations, bounded
Writer accounting, SHA-256 seals, historical Git-blob verification and frozen
numerical reductions are reused. The normal runtime, CPU LoRA, cache, generation
and old independent analyzers remain unchanged.

## Historical evidence and sealed plan

Preparation requires explicit original run, exported arithmetic review, expected
historical commit and base preparation paths. It verifies:

* COMPLETE raw block status and its complete SHA-256 inventory;
* original sealed plan/configuration/source-inventory chain;
* every recorded historical source SHA against its Git blob;
* exported independent audit PASS and real GPU evidence completeness;
* one real Qasper Adapter_A/User_01 S64/F256 token-prefix pair;
* first 64 source/full projection-input bytes equal for each Q/K/V role;
* prefill positions, masks, lengths and recorded S/F backend configuration;
* prepared base/model/tokenizer identity and offline tokenizer/configuration.

Missing objects, inputs, altered seals or inconsistent references fail closed.
This is verification of historical exports and original sealed bytes, not a fresh
execution of the old independent GPU audit under changed source. The recorded
historical commit stays separate from the current reviewed diagnostic commit.

Only observed values are copied. If the full 256-row input is not captured, its
missing 192 rows use `SYNTHETIC_XORSHIFT32_DYADIC_V1`, seed 42: integer xorshift32,
then `((state & 2047)-1024)/1024`. These finite dyadic values are exactly
representable in FP16. The historical prefix is unchanged. The plan explicitly
labels each role's suffix method, preserves historical tensor/layout metadata,
and records token/request identities and the original run/review seals.

This is a controlled shape extension, not a production-forward replay. Available
historical base outputs are compared descriptively; full-context, allocator
alignment and scheduling equivalence are not assumed. The analyzer deliberately
does not assert HISTORICAL_DIVERGENCE_REPRODUCED from similar error magnitudes.

## One GPU diagnostic

One compatible RTX 3090, one bare OPT-2.7B FP16 load, Layer 0 q/k/v only. Native
FP16 calls the original `torch.nn.Linear.forward` on the actual loaded projection
module, with original bias and weights. PEFT wrappers, overridden forward methods
and projection hooks are rejected. The loaded model's full fingerprint and each
projection's saved parameter hashes must match preparation. No normal model
forward, LoRA correction, TCP frame, cache access or generation occurs.

Two contiguous tensors have shapes `[1,64,2560]` and `[1,256,2560]`. Immediate
pre/post-call receipts check full input hashes, aligned prefix hashes, original
parameters, active parameters, execution UUID and effective backend settings.
Each call synchronizes CUDA before execution and after its output. Initial and
final complete model fingerprints must match. Diagnostic timing is not serving
performance evidence.

Default: two warmups per shape, then six measurements per shape. Alternating
rounds execute 64→256 and 256→64, three rounds each. Eight repetitions are also
bounded and supported. FP32 repeats the same schedule using exact numeric
promotions of inputs and temporary copies of only the current selected projection
weight/bias. It calls `F.linear`; the production model is never cast or replaced.
The independent CPU audit checks those FP32 parameter hashes against exact
promotions of the saved native FP16 parameters. It does not treat FP32 as ground
truth for the complete model.

The historical core backend is an explicitly declared baseline condition:
`EXPLICIT_REPLAY_RECORDED_CORE_SETTINGS`. Preparation prints it; execution records
initial/effective/final settings. TF32, FP16 reduction, float32 precision,
deterministic-algorithm and cuDNN flags/thread count are replayed, not tuned to seek
equality. The recorded workspace environment must be set before Python. Library
versions must match preparation and the recorded Torch/Transformers/CUDA versions.
Publicly unavailable cuBLAS version information is recorded UNKNOWN. No alternate
backend sweep is performed.

Contiguous input is the primary control. If historical observations require a
noncontiguous or offset view, an additional bounded `HISTORICAL_STRIDED` control
recreates recorded strides/offsets using independent storage and identical numeric
values. Overlapping or oversized layouts fail rather than silently normalize.
Only recorded layout is reproduced; allocator address alignment is unknown.
Matched layout comparisons remain separate from the primary shape verdict.

Resource bounds: four CPU cores, 64 GiB Slurm RAM, 32 GiB available CPU reserve,
20-minute job, 900-second operation deadline, one runtime-verified RTX 3090 and
more than `5.4 GB + 5 GiB` free GPU memory before loading. FP32 allocation also
checks available memory before creating the selected copies. Evidence is bounded
to 512 MiB including a 64 MiB accounting reserve and 256 data files, at most
32 MiB per tensor. Default contiguous tensor data is approximately 136 MiB;
eight repetitions plus the optional layout remain below the cap. Require over
2 GiB disk headroom. Outputs include all rows of each measured result, enabling
full 256-row repeatability checks; attention tensors are never dumped.

## Independent analysis and verdicts

All same-shape pairs are compared across all output rows. Cross-shape comparisons
align the first 64 positions within each balanced round. Results include exact
byte hashes, bitwise equality, absolute errors, relative L2, differing values,
frozen elementwise/max/relative verdicts and finite FP16 ULP distances. Signed
zeros have ULP distance zero; unexpected NaN/Inf are rejected. Existing thresholds
are imported unchanged. These activation checks do not substitute for logit,
generation or full-system fidelity gates.

Stable FP16 repeat outputs plus cross-shape differences can demonstrate behavior
of this actual operation under the declared contiguous control. Unstable repeats
confound that inference. FP32 equality can support precision-sensitive behavior;
FP32 differences require further investigation. Equal isolated outputs mean the
original difference was not reproduced here, not that shape effects are impossible.
Per-role and per-layout verdicts, historical comparisons and all error records are
kept separate. A strided compound control cannot establish the primary contiguous
shape verdict. No case confirms a cuBLAS algorithm, accumulation order, cache
cause, SemCache speedup or restored integration fidelity.

## Local CPU commands and delivery

```bash
cd /home/songsong/SemCache
PYTHONPATH=src:tests python3 -m unittest \
  test_r5_i5r2bs1 test_r5_i5r2b test_r5_i5r2 test_r5_i5r1 -v
bash -n jobs/r5_i5r2bs1_linear.sh
python3 scripts/133_prepare_r5_i5r2bs1.py --help
python3 scripts/134_run_r5_i5r2bs1.py --help
python3 scripts/135_analyze_r5_i5r2bs1.py --help
```

Local scalar fixture perturbations are deliberately injected and never GPU results.
The actual small Torch helper test is NOT_RUN if Torch/NumPy are absent. No model
or dataset is downloaded. All nine new source/job/test/documentation files must be
reviewed and committed together; preparation refuses untracked execution sources
or a dirty tracked tree. The user handles commit/push/deployment.

Final local validation: 105 tests executed, **100 PASS / 0 FAIL / 5 SKIP**.
S1: 22 PASS / 1 SKIP; R2B: 27 PASS / 4 SKIP; R2A: 23 PASS; R1: 20 PASS;
R2B/R1 process/TCP regressions: 8 PASS. The skips are the actual S1 Torch helper,
the installed R2B observer API, two actual R2B CPU tensor tests and actual OPT
hook test: Torch/NumPy/Transformers are absent locally. They are NOT_RUN, not GPU
verification. Initial sandbox socket restrictions required a socket-enabled CPU
rerun. All process tests and their independent fixture audits passed. Shell and
Python syntax and the three CLI help commands passed.

The synthetic S1 plan/run/audit is retained in
`results/r5/i5r/numerical_fidelity/linear_shape_control/cpu_verification_01`.
Its known injected difference is explicitly CPU_FIXTURE_ONLY. The existing raw
job 152846 was not accessible or newly reproduced on this workstation.

Exact local staging instructions, excluding results/logs/unrelated files:

```bash
git add src/semcache/edgelora/r5_i5r2bs1_plan.py \
  src/semcache/edgelora/r5_i5r2bs1_runtime.py \
  src/semcache/edgelora/r5_i5r2bs1_analysis.py \
  scripts/133_prepare_r5_i5r2bs1.py scripts/134_run_r5_i5r2bs1.py \
  scripts/135_analyze_r5_i5r2bs1.py jobs/r5_i5r2bs1_linear.sh \
  tests/test_r5_i5r2bs1.py docs/R5_I5R2B_S1_LINEAR_SHAPE_CONTROL.md
git diff --cached --stat
git diff --cached --check
git commit -m "Add isolated R5-I5R-2B-S1 native Linear shape diagnostic"
```

No commit, deployment, push, Slurm submission or GPU run was performed by the
implementation agent. The implementation is pending user review/commit.

## SERAPH CPU preparation after user-managed pull

Keep the reviewed source unchanged until independent analysis completes. Select
the actual original run and exported review paths; only their basenames are known
here. Do not select an audit directory as the raw run.

```bash
cd /data/khuss/repos/GP_semcache
test "$(git branch --show-current)" = exp/raw-semcache-r5-integration
git pull --ff-only
conda activate semcache
export I5_CONDA_ROOT="$(conda info --base)"
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
read -r -p 'Reviewed S1 source commit: ' S1_SOURCE_COMMIT
export S1_SOURCE_COMMIT
test "$(git rev-parse HEAD)" = "$S1_SOURCE_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
read -r -e -p 'Original block_0_152846 directory: ' S1_HISTORICAL_RUN
read -r -e -p 'block_0_152846_arithmetic_review_02 directory: ' S1_HISTORICAL_REVIEW
export S1_HISTORICAL_RUN S1_HISTORICAL_REVIEW
export S1_BASE_PREPARED="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_prepared"])' "$S1_HISTORICAL_RUN/i5_execution_config.json")"
export S1_HISTORICAL_COMMIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$S1_HISTORICAL_RUN/i5_execution_config.json")"
export S1_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export S1_EXPERIMENT="$PWD/results/r5/i5r/numerical_fidelity/linear_shape_control/plan_$S1_TAG"
export S1_RUN_ROOT="$PWD/results/r5/i5r/numerical_fidelity/linear_shape_control/attempt_$S1_TAG"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python -m unittest test_r5_i5r2bs1 -v
python scripts/133_prepare_r5_i5r2bs1.py \
  --historical-run-dir "$S1_HISTORICAL_RUN" --historical-review-dir "$S1_HISTORICAL_REVIEW" \
  --expected-historical-source-commit "$S1_HISTORICAL_COMMIT" \
  --base-prepared "$S1_BASE_PREPARED" --output-dir "$S1_EXPERIMENT" --repetitions 6 --suffix-seed 42
```

Review the printed suffix methods, source commit, bounds and historical backend.
The commands derive paths from metadata, but preparation independently verifies
all inventories and historical Git blobs; metadata is never trusted without those
checks. Real preparation needs the existing compatible Torch/Transformers/PEFT/
NumPy environment and offline checkpoints, not a new download.

## Exactly one manual submission

Choose a node permitted by the allocation policy. There is no fixed node/default
UUID in the job. Both moana-y3 and moana-y4 are permitted choices; actual hardware
is still checked at runtime. Every mandatory environment variable below is
validated before loading the model. Arrays are rejected.

```bash
read -r -p 'Permitted GPU node (moana-y3 or moana-y4): ' S1_NODE
export S1_NODE S1_ENABLE_GPU=1
export CUBLAS_WORKSPACE_CONFIG="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["historical_backend"]["CUBLAS_WORKSPACE_CONFIG"])' "$S1_EXPERIMENT/i5r2bs1_plan.json")"
mkdir -p logs
S1_JOB_ID="$(sbatch --parsable --export=ALL --nodelist="$S1_NODE" jobs/r5_i5r2bs1_linear.sh)"
export S1_JOB_ID
```

This is an operator command only; no Python program submits jobs. Do not reuse a
failed output directory. Retain plan, input, node, commit and run-root variables
for review; restore the exact recorded values if the shell is restarted.

## CPU audit after completion

```bash
export S1_RUN="$S1_RUN_ROOT/run_${S1_JOB_ID%%;*}"
python -m json.tool "$S1_RUN/i5r2bs1_status.json"
CUDA_VISIBLE_DEVICES='' python scripts/135_analyze_r5_i5r2bs1.py \
  --experiment "$S1_EXPERIMENT" --run-dir "$S1_RUN" --output-dir "${S1_RUN}_audit_01"
```

Plan: `i5r2bs1_plan.json`, selected input/reference blobs and `i5_output_manifest.json`.
Run: `i5r2bs1_records.json`, `i5r2bs1_weights.json`, `i5r2bs1_inputs.json`,
`i5r2bs1_backend.json`, `i5r2bs1_base_fingerprint.json`, `i5r2bs1_status.json`,
selected weights/input/output binaries, Writer accounting and the seal.
Audit: `i5r2bs1_comparisons.json`, `i5r2bs1_verdicts.json`,
`i5r2bs1_independent_audit.json`, `i5r2bs1_source_provenance.json`,
`i5r2bs1_report.md` and `i5_output_manifest.json`.

The next smallest experiment is conditional on these results: repeat variability
requires execution-state/nondeterminism investigation; stable FP16-only shape
effects justify a separately reviewed kernel/backend trace; stable differences
in both precisions require layout/backend scrutiny; equality here calls for an
original-layout/runtime-context control. None is automatically prepared or run.
