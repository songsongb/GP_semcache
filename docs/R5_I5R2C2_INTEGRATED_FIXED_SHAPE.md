# R5-I5R-2C-2: integrated native-only projection padding

Experimental opt-in only. Normal inference and all frozen numerical thresholds
remain unchanged. No GPU findings are available from local CPU fixtures.

The three independently sealed plans OFF / SOURCE_ONLY / SOURCE_AND_SUFFIX reuse
R2B preparation, R1's exact Qasper 64/256-token pair, the thirty-user/three-adapter
registry and two-token target budget (EOS can end earlier). Only Adapter_A/User_01
has measured requests. All A/B/C conditions execute normally with fresh model,
UD sessions, semantic index and physical GPU cache for each condition and mode.
Existing warmups remain. Each condition uses the existing runtime, TinyBERT,
clusters=1, LRU, CPU LoRA and actual localhost TCP. No HIT is forced.

The shared bridge is deliberately small: physical_path reads an optional session
callback for native Linear only; QualityDriver binds that callback only when an
experimental driver supplies one. With no callback, the original delta-then-base
then addition/cast arithmetic is unchanged. The old runner dispatches C2 only for
a sealed C2 plan. The output safety helper allows the new narrow namespace.

A/B and C/OFF install no transform. SOURCE_ONLY pads the standalone C source
native input by appending zeros (64→256), then slices output 0:64. SOURCE_AND_SUFFIX
also prepends 64 zeros to actual HIT suffix native input (192→256), retaining
64:256. Gates check condition, request ID, prefill step, cache grant, mode and
real input/context lengths. Decode and warmup never pad. All 32 layers and Q/K/V
roles use the selected path. No token IDs, masks, positions, other module inputs,
LoRA tensors, persistent KV or cached Q consumption policy are changed. CPU LoRA
executes first on real rows. Actual K/V prefix consumption remains required.

Padding receipts are distinct from legacy real-row accounting: a source adds
192 rows per Q/K/V call; padded suffix adds 64. Extra FLOPs and conservative
live temporary input/output/cropped byte bounds are separate. Legacy skipped
prefix rows describe omitted **real** rows; they must not be presented as net
native computation savings when dummy rows were computed. Target native GEMM
work may reach full-target row count while KV reuse/LoRA/TCP savings still exist.
No speedup or production feasibility is claimed.

The bounded evidence reuses R2B Layer-0 capture (32 MiB/128 tensor files per
endpoint), normal all-layer cache/shape/hash/TCP proofs and full-vocabulary logits.
Additional Layer-0 final-real-prefill attention/layer outputs are small. This is
not a full all-layer tensor dump or native PEFT parity proof. Existing independent
I5 audit is retained; C2 additionally checks native padding receipts and Layer-0
saved-versus-consumed KV bytes. Numerical failure is preserved separately from
functional/audit completion. Two-token generation budget is not proof that two
steps actually executed; EOS and per-step skipped-history fields remain visible.
Every padding receipt binds its real-region hash/shape to the actual ES HIDDEN
TCP ledger and verifies dummy-region bytes are zero. Layer-0 source/full-target
prefix comparisons include available normalization, projection inputs and
base/delta/raw QKV; final-real-position attention/layer outputs are additional
observations, not proof of causality. Priming-source and scored-target numerical
verdicts are separate; overall strict restoration requires both, including
generated token/text and task-quality checks. Native PEFT parity remains
UNESTABLISHED. Missing captures remain unavailable rather than presumed equal.

The job uses one GPU at a time, four cores, 64 GiB RAM and one hour, with a
1000-second bound per mode. Nine sequential model loads follow from the inherited
fresh-process A/B/C isolation (three conditions × three modes), not multiple live
model copies. Sharing a live model across those independent contracts is not
introduced here. Setup/warmup remain outside request timing. Source-inclusive
cost sums real source and target request intervals including admission; request
intervals exclude inter-request/startup/warmup overhead. All timing is diagnostic,
including tensor-copy/hash/evidence interference. Source cost can be amortized
only in a separately measured repeated-HIT workload.

## Operator preparation

After reviewing and manually committing/pushing/pulling all changed files, use
fresh paths. Historical R1, R2A and completed C1 evidence must remain available.
C1 plan/run/audit inventories and historical Git blobs are verified. C1's original
S1 pair and base identity must match the R1 controlled pair.

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export I5_CONDA_ROOT="$(conda info --base)"
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
read -r -p 'Reviewed C2 commit: ' C2_SOURCE_COMMIT
export C2_SOURCE_COMMIT
test "$(git rev-parse HEAD)" = "$C2_SOURCE_COMMIT"
read -r -e -p 'Original R1 smoke plan: ' R1_PLAN
read -r -e -p 'Completed R2A review: ' R2A_REVIEW
read -r -e -p 'Sealed C1 plan: ' C1_PLAN
read -r -e -p 'Completed C1 run: ' C1_RUN
read -r -e -p 'Independent C1 audit: ' C1_AUDIT
export C2_MULTI_PREPARED="$PWD/results/r5/multiuser/expanded_152381_20261031T154848Z/adapters"
export C2_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export C2_EXPERIMENT="$PWD/results/r5/i5r/numerical_fidelity/integrated_fixed_shape/plan_$C2_TAG"
export C2_RUN_ROOT="$PWD/results/r5/i5r/numerical_fidelity/integrated_fixed_shape/attempt_$C2_TAG"
python scripts/138_r5_i5r2c2.py prepare --historical-plan "$R1_PLAN" \
  --multi-prepared "$C2_MULTI_PREPARED" --r2a-review "$R2A_REVIEW" \
  --c1-plan "$C1_PLAN" --c1-run "$C1_RUN" --c1-audit "$C1_AUDIT" \
  --output-dir "$C2_EXPERIMENT"
```

One manual job, no array or automatic expansion:

```bash
read -r -p 'Permitted node (moana-y3 or moana-y4): ' C2_NODE
export C2_NODE C2_ENABLE_GPU=1
mkdir -p logs
C2_JOB_ID="$(sbatch --parsable --export=ALL --nodelist="$C2_NODE" jobs/r5_i5r2c2_integrated.sh)"
export C2_JOB_ID
```

The inherited I5 launcher explicitly sets its frozen CUDA workspace/environment
and repeats offline model/tokenizer, distinct adapter, TinyBERT, installed OPT,
GPU UUID/class/memory, CPU, disk, source and allocation checks. Missing required
environment fails before inference. Never reuse a failed output directory.

CPU independent analysis with the same pinned source after completion:

```bash
export C2_RUN="$C2_RUN_ROOT/run_${C2_JOB_ID%%;*}"
CUDA_VISIBLE_DEVICES='' python scripts/138_r5_i5r2c2.py analyze \
  --experiment "$C2_EXPERIMENT" --run-dir "$C2_RUN" \
  --output-dir "${C2_RUN}_review_01"
```

Each mode retains its sealed raw run and ordinary I5 audit. C2 reports are
`i5r2c2_numerical.json`, `i5r2c2_layer0.json`, `i5r2c2_accounting.json`,
`i5r2c2_decision.json` and `i5_output_manifest.json`. Physical reuse, strict
fidelity, native PEFT parity and performance status remain separate. Numerical
failure does not imply a broken cache, and a physical HIT does not imply fidelity.

## CPU validation

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r2c2 -v
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r1 test_r5_i5r2b test_r5_i5r2c1 -v
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r2c2_processes -v
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r1_processes -v
bash -n jobs/r5_i5r2c2_integrated.sh
```

TCP fixtures require localhost socket permission. They are never GPU evidence.
Synthetic archived-source helpers now record committed Git blob hashes rather
than dirty working-tree hashes under HEAD; strict real provenance is unchanged.
No historical result is edited. All changes remain pending user review/commit.

Final local validation: **68 PASS / 0 FAIL / 5 SKIP** (73 tests): focused and
R1/R2B/C1 unit regressions 60 PASS/5 SKIP, three C2 subprocess/TCP tests PASS,
five R1 subprocess regressions PASS. Skips require unavailable local
PyTorch/NumPy/Transformers dependencies, including the real CPU FP16 padding API
test. Scalar CPU fixtures do not validate real GPU arithmetic. The independent
CPU analysis CLI also completed for all three modes, reporting CPU_FIXTURE_ONLY.
Artifacts are preserved under
`results/r5/i5r/numerical_fidelity/integrated_fixed_shape/cpu_verified_final_03/`.
GPU_NOT_RUN; physical GPU reuse and strict FP16 fidelity remain unverified.

Review and stage exactly these delivery files before manually committing; avoid
staging unrelated results/logs or historical artifacts:

```bash
git add scripts/119_run_r5_i5.py scripts/138_r5_i5r2c2.py \
  src/semcache/edgelora/r3b_runtime.py src/semcache/edgelora/r5_i5_driver.py \
  src/semcache/edgelora/r5_i5_evidence.py src/semcache/edgelora/r5_i5r2c2.py \
  tests/test_r5_i5r2bs1.py tests/test_r5_i5r2c0.py \
  tests/test_r5_i5r2c2.py tests/test_r5_i5r2c2_processes.py \
  jobs/r5_i5r2c2_integrated.sh docs/R5_I5R2C2_INTEGRATED_FIXED_SHAPE.md
git diff --cached --check
git diff --cached --stat
git commit -m 'Add opt-in integrated fixed-shape FP16 diagnostic'
```
