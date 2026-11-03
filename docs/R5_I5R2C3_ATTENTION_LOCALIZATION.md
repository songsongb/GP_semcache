# R5-I5R-2C-3: Layer-0 attention localization

DIAGNOSTIC_ONLY. No production file, numerical threshold, LoRA/TCP/cache path,
Full Recompute reference, or historical result is modified. GPU_NOT_RUN locally.
Job 152894 findings are the operator's verified handoff, not locally reproduced.

## Evidence-first design and blocking condition

Stage A reads **one mode** of the original C2 plan/run and its exported C2 audit.
Start with SOURCE_ONLY; SOURCE_AND_SUFFIX and OFF can be reviewed separately in
fresh outputs. It verifies complete SHA-256 inventories, original Git blobs,
plan/configuration/audit chaining, actual source64/full256 contexts, model and
adapter identities, recorded backend equality, tensor references and masks.
It never runs the old independent GPU analyzer under a changed source snapshot.
Historical exported audit verification and new read-only checks remain distinct.

The C2 preparation sets task `strict=False`; ordinary scalable QKV/component
records can therefore be hash-only. R2B diagnostic tensors retain just positions
0:64. A full-target additive mask capture retains 64 queries × 256 keys. These
are enough to check prefix equality and causal visibility, but **not** to replay
native attention at 256 queries with the actual missing Q/K/V suffix values.
Stage A detects what is actually present, rather than assuming those values were
saved. It lists each missing operand explicitly. No suffix, mask, position,
hidden state, or historical tensor is synthesized.

If these operands are absent in job 152894, Stage B preparation **must fail**.
Do not submit a job. The next smallest action is to locate complete original
sealed values elsewhere in that run. If they were never captured, a separately
reviewed bounded S64/F256 Layer-0 operand capture is needed; it is not implemented
as an automatic fallback and must not rerun nine conditions by default. This
stage cannot recover missing historical values from hashes.

`--allow-incomplete` explicitly permits a partial Stage A coverage report. Without
it, missing replay operands produce a sealed report and exit 2. Corruption,
wrong source identity, missing required core manifests/contexts or incompatible
masks always fail; partial mode never bypasses those gates.

## Conditional native replay

Replay preparation succeeds only with complete, verified recorded Q/K/V at
64/256 rows, actual complete additive masks and historical module outputs.
It copies bounded operands into a fresh sealed plan and pins the reviewed source.
The single-model GPU program calls the actual installed eager OPTAttention,
whose forward source hash must equal the historical contract. Temporary
projection wrappers return the recorded **total** Q/K/V; no Linear/LoRA/TCP,
cache access, model forward or generation is executed. Recorded Q is used only
as the forward's shape carrier, not represented as a recovered hidden state.
Original out_proj weights and bias are verified against the loaded full model
fingerprint. All patches/hooks are removed even on failure. Production is untouched.

Native TorchFunctionMode observations capture scaled QK BMM output, actual
softmax input (after mask/clamp), actual probabilities passed to PV BMM, PV BMM
output, output-projection input/output and final module output. There is no
manually substituted attention arithmetic. Unsupported operation counts/backend
fail closed. No claim can distinguish mask-add versus subsequent clamp when only
the combined softmax input is observed. Masks and zero future probabilities are
checked before shared 64-query/64-key comparisons. Full-target extra keys are
retained for causal visibility checks. BMM input prefix hashes/layouts distinguish
upstream operand differences from a differing BMM output.

Source/full order alternates, two warmup calls per shape and six measured calls
per shape. Only first64 query output rows are exported. Original tensor metadata
is retained; layouts are reconstructed only with complete compatible recorded
stride/offset metadata, otherwise explicitly labeled contiguous controls.
Exact historical module-prefix replay at **every repetition of both shapes** is
mandatory, as is same-shape repeatability. A failed replay is sealed and analyzed
but exits nonzero and cannot localize the historical divergence.

One compatible RTX3090, four cores, 64 GiB RAM, 30-minute Slurm limit, 600-second
in-program bound, 256 MiB evidence budget and 192 data-file bound. One model load;
no production latency, E2E or speedup claim. Backend settings are explicitly
restored to recorded values and effective settings/versions verified; unavailable
or mismatched settings reject the run. An actual different compatible GPU UUID
is recorded, never forged to equal historical preparation.

## Operator commands: Stage A (CPU only, first)

Manually review/commit/push/pull the six new files first. No remote deployment is
performed by the implementation. Do not edit or reseal historical outputs.

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export I5_CONDA_ROOT="$(conda info --base)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
read -r -e -p 'Original SOURCE_ONLY mode plan directory: ' C2_PLAN
read -r -e -p 'Job 152894 SOURCE_ONLY raw mode directory: ' C2_RUN
read -r -e -p 'Corresponding exported C2 independent audit directory: ' C2_AUDIT
read -r -p 'Verified historical C2 source commit: ' C2_COMMIT
export C3_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export C3_ROOT="$PWD/results/r5/i5r/numerical_fidelity/attention_localization"
CUDA_VISIBLE_DEVICES='' python scripts/139_r5_i5r2c3.py stage-a \
 --experiment "$C2_PLAN" --run-dir "$C2_RUN" --audit-dir "$C2_AUDIT" \
 --expected-historical-commit "$C2_COMMIT" \
 --output-dir "$C3_ROOT/review_$C3_TAG" --allow-incomplete
```

Inspect `i5r2c3_verdicts.json`, especially `GPU_REPLAY_READY` and `missing`.
Stop if false. Keep the old Git objects available; no unavailable-source bypass
exists. Required paths are prompted because the real job152894 directory names
and source SHA were not supplied locally. Do not guess them.

## Stage B: only when the recorded operands are complete

```bash
read -r -e -p 'Validated offline base_prepare directory: ' C3_BASE_PREPARED
read -r -p 'Reviewed current C3 commit: ' C3_SOURCE_COMMIT
export C3_SOURCE_COMMIT
 test "$(git rev-parse HEAD)" = "$C3_SOURCE_COMMIT"
export C3_EXPERIMENT="$C3_ROOT/plan_$C3_TAG"
export C3_RUN_ROOT="$C3_ROOT/attempt_$C3_TAG"
python scripts/139_r5_i5r2c3.py prepare \
 --experiment "$C2_PLAN" --run-dir "$C2_RUN" --audit-dir "$C2_AUDIT" \
 --expected-historical-commit "$C2_COMMIT" \
 --base-prepared "$C3_BASE_PREPARED" --output-dir "$C3_EXPERIMENT"
```

Only after preparation succeeds, set `CUBLAS_WORKSPACE_CONFIG` exactly as recorded
in Stage A's backend receipt **before Python starts**. If it was null, unset it:
`unset CUBLAS_WORKSPACE_CONFIG`; otherwise export that recorded value. The run
rejects a mismatch. No pass-seeking backend change is permitted.

```bash
read -r -p 'Permitted node (moana-y3 or moana-y4): ' C3_NODE
export C3_NODE C3_ENABLE_GPU=1
mkdir -p logs
test -f "$C3_EXPERIMENT/i5r2c3_plan.json"
C3_JOB_ID="$(sbatch --parsable --export=ALL --nodelist="$C3_NODE" jobs/r5_i5r2c3_attention.sh)"
export C3_JOB_ID
```

CPU-only independent analysis after completion, same reviewed source:

```bash
export C3_RUN="$C3_RUN_ROOT/run_${C3_JOB_ID%%;*}"
CUDA_VISIBLE_DEVICES='' python scripts/139_r5_i5r2c3.py analyze \
 --experiment "$C3_EXPERIMENT" --run-dir "$C3_RUN" \
 --output-dir "${C3_RUN}_audit_01"
```

## Artifacts and interpretation

Stage A: `i5r2c3_evidence_inventory.json`, `i5r2c3_tensor_comparisons.json`,
`i5r2c3_source_provenance.json`, `i5r2c3_verdicts.json`, `i5r2c3_report.md`,
`i5_output_manifest.json`. Replay plan: `i5r2c3_plan.json` and copied inputs.
Run: plan, records, backend/layout receipt, weights/base fingerprint, status,
bounded tensor binaries, SHA-256 inventory. Independent replay audit: comparisons
(including repetitions and exact historical replay), verdicts, audit and report.

Verdicts are separate: EVIDENCE_VALIDATED, HISTORICAL_REPLAY_REPRODUCED,
FIRST_DIVERGENCE_LOCALIZED, REPLAY_NOT_ESTABLISHED, INCOMPLETE_EVIDENCE.
PHYSICAL_CACHE_REUSE is HISTORICAL_C2_VERIFIED, not a fresh test.
STRICT_NUMERICAL_FIDELITY_RESTORED is NOT_EVALUATED; ROOT_CAUSE_CONFIRMED is false.
A first unequal native boundary does not identify a particular kernel/rounding
mechanism. Different score operands prevent attribution to QK BMM itself.
Later residual/norm/FFN operations are not replayed; their potential contribution
is unbounded by an attention-only result. Missing internal values remain unknown.

## Local CPU validation

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r2c3 -v
PYTHONPATH=src:tests python3 -m unittest \
 test_r5_i5r2c2 test_r5_i5r2bs1 test_r5_i5r2b test_r5_i5r2c1 -v
bash -n jobs/r5_i5r2c3_attention.sh
python3 scripts/139_r5_i5r2c3.py --help
```

Synthetic fixture checks establish alignment, mask rules, evidence integrity and
report logic only. Installed native OPTAttention CPU API testing is skipped when
PyTorch/NumPy/Transformers are unavailable; no GPU numerical evidence is inferred.

Final local execution: **81 PASS / 0 FAIL / 7 SKIP** across 88 tests: C3 focused
19 PASS/1 SKIP, C2/S1/R2B/C1 regressions 62 PASS/6 SKIP. The seven skips explicitly
require unavailable Torch/NumPy/Transformers APIs. Shell syntax, Python AST and
all CLI subcommand help checks passed. No GPU or SERAPH evidence was read locally.
Only new synthetic fixtures supplied historical tensor values for local testing.
An attempt to review an older local C2 CPU artifact was correctly rejected by
historical Git SHA checks because it recorded pre-commit source bytes; its
manifests were neither changed nor bypassed. No real job152894 values or results
are reproduced by these CPU checks.

All six delivery files are additive and await user review/commit:
`src/semcache/edgelora/r5_i5r2c3.py`,
`src/semcache/edgelora/r5_i5r2c3_runtime.py`,
`scripts/139_r5_i5r2c3.py`, `jobs/r5_i5r2c3_attention.sh`,
`tests/test_r5_i5r2c3.py`, and this document.
