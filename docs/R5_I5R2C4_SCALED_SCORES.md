# C4 scaled-score isolation and bounded I5R-2 assessment

DIAGNOSTIC_ONLY. No production files, serializers, fingerprints, cache policy,
weights, attention arithmetic or numerical thresholds change. Historical jobs
C2 152894, B0 153199 and C3 153201 are operator-verified handoffs, not locally
reproduced GPU findings. Every real source path and expected historical commit is
explicit. No historical inventory is overwritten/resealed; no job is auto-submitted.

## Matrix and interpretation

All inputs are actual native **already-scaled** Q and transposed K recovered from
C3's verified B0 raw operands by the installed eager OPTAttention implementation:

| Case | Q shape | KT shape | Interpretation |
|---|---|---|---|
| A | [1,32,64,80] | [1,32,80,64] | Native source baseline |
| B | [1,32,64,80] | [1,32,80,256] | Key-length control |
| C | [1,32,256,80] | [1,32,80,64] | Query-length, **noncausal** diagnostic |
| D | [1,32,256,80] | [1,32,80,256] | Native full baseline |

Each case runs NATIVE recorded strides and CONTIGUOUS materialized operands.
Each layout runs FP16 and FP32 matmul; every FP32 result also supplies a verified
FP16 cast. This gives 24 output conditions, 6 measured repetitions each (144 common
region tensors), 96 measured matmuls and 32 warmup matmuls. Two warmups per operator
condition precede measurements; forward/reverse condition order alternates. Compare
only the common [32,64,64] region. Query-length contrasts are AC/BD; key-length
contrasts AB/CD; combined AD. Repetitions share one prompt and are not independent
dataset groups or grounds for confidence intervals/general speedup claims.

No synthetic suffix, hidden state, mask or scaling replacement is permitted.
Recovery runs only two Layer0 attention calls (no full-model forward), after one
OPT-2.7B FP16 load/fingerprint verification. It checks reconstructed QK full/prefix
hashes and layout against **all** C3 repetitions. Native source/full attention
scores/module prefixes and standalone FP16 matmul baselines must reproduce C3
bitwise before controls begin. If native post-QK scaling is non-unit and cannot
be represented by those operands, the direct matmul baseline will fail: stop,
preserve hashes/tensors, and do not silently move scaling to Q or change its order.

The same scaled operands are promoted for FP32. Weights/model are never cast.
No backend overrides beyond the explicitly declared recorded baseline replay are
allowed; actual TF32, reduced-precision and deterministic settings remain recorded.
FP32 here means FP32 operands under recorded backend flags, not an assumed exact
reference. Native strides/offsets are restored; contiguous controls allocate real
materialized copies with equal values. Allocation alignment modulo256 is recorded.
A layout-associated difference supports layout sensitivity, not a proven isolated
CUDA algorithm or accumulation mechanism. If no distinct layout exists, report
UNSUPPORTED_NO_DISTINCT_LAYOUT instead of an informative negative layout test.

## Reproduction/audit gates

Preparation verifies the sealed C2 plan/run/audit, B0 plan/run/audit and C3
plan/run/audit, expected historical Git commits/all recorded blobs, model and
adapter identity, C3 companion binding, GPU class, backend receipts and all tensor
references. It uses archived audit verification plus new CPU checks, never disables
the existing C3 source guard or pretends to rerun its historical GPU audit from a
new source checkout. The new reviewed C4 commit is separately pinned.

Execution uses one compatible RTX3090,4 cores,64 GiB RAM,900-second program bound,
30-minute Slurm allocation,256 MiB evidence/accounting budget and256 data-file
bound. No generation, CPU UD, semantic search or cache operation is run in C4.
Those are historical B0/C2 inputs. Native recovery uses actual installed attention
and native recorded QKV; isolated controls use actual torch.matmul. These are
separate scopes, never an end-to-end benchmark.

Independent CPU analysis rechecks inventories, all input provenance, recovered
prefix hashes, exact native layout, baseline/module reproduction, all four shapes,
layout identities, exact FP32 promotions, output rounding, complete repetitions,
weight immutability and GPU/backend receipts. Native repeated A/D must remain
bitwise equal to the historical baseline. Unstable/incomplete controls cannot
support attribution. Failed native recovery/baselines seal available evidence and
exit nonzero. A sealed failed run can receive a FAIL audit and bounded assessment
with C4 controlled findings NOT_EVALUATED. Corruption/provenance mismatches reject.
No retry or expanded sweep is automatic.

## SERAPH preparation (CPU only)

Manually review/commit/push/pull the six new files first, keeping historical Git
objects and all old outputs intact. The supplied job IDs do not define their
filesystem paths: enter the actual sealed directories, not guessed names.

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export I5_CONDA_ROOT="$(conda info --base)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for c4_stage in C2 B0 C3; do
  for c4_role in EXPERIMENT RUN AUDIT; do
    read -r -e -p "Verified $c4_stage $c4_role directory: " "C4_${c4_stage}_${c4_role}"
  done
  read -r -p "Verified $c4_stage historical source commit: " "C4_${c4_stage}_COMMIT"
done
read -r -e -p 'Original validated base_prepare directory: ' C4_BASE_PREPARED
read -r -p 'Reviewed current C4 source commit: ' C4_SOURCE_COMMIT
export C4_SOURCE_COMMIT
test "$(git rev-parse HEAD)" = "$C4_SOURCE_COMMIT"
export C4_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export C4_ROOT="$PWD/results/r5/i5r/numerical_fidelity/scaled_score_isolation"
export C4_EXPERIMENT="$C4_ROOT/plan_$C4_TAG"
export C4_RUN_ROOT="$C4_ROOT/attempt_$C4_TAG"
CUDA_VISIBLE_DEVICES='' python scripts/141_r5_i5r2c4.py prepare \
 --c2-experiment "$C4_C2_EXPERIMENT" --c2-run "$C4_C2_RUN" --c2-audit "$C4_C2_AUDIT" \
 --b0-experiment "$C4_B0_EXPERIMENT" --b0-run "$C4_B0_RUN" --b0-audit "$C4_B0_AUDIT" \
 --c3-experiment "$C4_C3_EXPERIMENT" --c3-run "$C4_C3_RUN" --c3-audit "$C4_C3_AUDIT" \
 --expected-c2-commit "$C4_C2_COMMIT" --expected-b0-commit "$C4_B0_COMMIT" \
 --expected-c3-commit "$C4_C3_COMMIT" --base-prepared "$C4_BASE_PREPARED" \
 --output-dir "$C4_EXPERIMENT"
```

Only after successful preparation, one manual GPU submission:

```bash
read -r -p 'Recorded CUBLAS_WORKSPACE_CONFIG (UNSET if null): ' C4_WORKSPACE
if [[ "$C4_WORKSPACE" == UNSET ]]; then unset CUBLAS_WORKSPACE_CONFIG; else export CUBLAS_WORKSPACE_CONFIG="$C4_WORKSPACE"; fi
read -r -p 'Permitted node (moana-y3 or moana-y4): ' C4_NODE
export C4_NODE C4_ENABLE_GPU=1
mkdir -p logs
C4_JOB_ID="$(sbatch --parsable --export=ALL --nodelist="$C4_NODE" jobs/r5_i5r2c4_scores.sh)"
export C4_JOB_ID
```

Independent CPU analysis and final assessment (fresh outputs):

```bash
export C4_RUN="$C4_RUN_ROOT/run_${C4_JOB_ID%%;*}"
export C4_AUDIT="${C4_RUN}_audit_01"
CUDA_VISIBLE_DEVICES='' python scripts/141_r5_i5r2c4.py analyze \
 --experiment "$C4_EXPERIMENT" --run-dir "$C4_RUN" --output-dir "$C4_AUDIT"
CUDA_VISIBLE_DEVICES='' python scripts/141_r5_i5r2c4.py final-assessment \
 --experiment "$C4_EXPERIMENT" --run-dir "$C4_RUN" --audit-dir "$C4_AUDIT" \
 --output-dir "${C4_RUN}_final_assessment_01"
```

Stop if provenance/inventory validation fails; do not force an assessment. A sealed
FAIL audit for a genuine incomplete/baseline-failed run permits only a bounded
assessment, never positive C4 causal findings.

## Artifacts and final-assessment schema

Plan: `i5r2c4_plan.json`. Run: plan, `i5r2c4_status.json`,
`i5r2c4_reproduction.json`, `i5r2c4_operands.json`, `i5r2c4_records.json`,
`i5r2c4_backend.json`, `i5r2c4_base_fingerprint.json`, bounded operand/baseline/
common-region tensor files, SHA-256 inventory. Audit:
`i5r2c4_comparisons.json`, `i5r2c4_independent_audit.json`, inventory.
Assessment: `i5r2_final_assessment.json` and `.md`, inventory.

Assessment schema I5R2_FINAL_BOUNDED_ASSESSMENT_V1 separates `observed_facts`
(historical reproduction, score localization, bitwise differences, frozen checks,
token/text/native task observations, historical physical reuse),
`c4_native_reproduction`, `controlled_findings` (shape/query/key/layout/precision),
`precision_shape_controls` (whether FP32 and its FP16 cast eliminate, reduce or
retain each cross-shape discrepancy), `hypotheses` (unresolved kernel/accumulation
mechanism), and `recommendations`. All same-shape repeat comparisons retain
their numerical metrics; they are not reduced to a single stability boolean.
Statuses are CONFIRMED, SUPPORTED_BUT_NOT_PROVEN, UNRESOLVED or NOT_EVALUATED.
Negative results and missing native quality data remain explicit. Score/attention
reproduction and operator sensitivity do not restore generation fidelity or
establish correctness-qualified speedup. ROOT_CAUSE_CONFIRMED remains false;
STRICT_NUMERICAL_FIDELITY_RESTORED is NOT_EVALUATED. This terminates I5R-2 at a
bounded operator-level conclusion unless a separately authorized correction and
integrated fidelity experiment follows.

## Local validation

Final local result: **88 tests, 85 PASS, 0 FAIL, 3 SKIP**. C4 focused:
16 tests, 15 PASS, 1 SKIP. C3/B0 regressions: 72 tests, 70 PASS, 2 SKIP.
The three skips require missing local Torch/NumPy/Transformers dependencies;
installed native APIs are exercised without skip-on-error handling when present.

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r2c4 -q
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r2c3 test_r5_i5r2c3b0 -q
bash -n jobs/r5_i5r2c4_scores.sh
python3 scripts/141_r5_i5r2c4.py --help
```

CPU virtual-array fixtures validate aggregation/control logic, not real matrix
arithmetic. The installed Torch CPU control test executes whenever Torch/NumPy
exist; it does not silently skip API failures. Local native tests are unavailable
when dependencies are absent. All C4 GPU experimental conclusions remain
NOT_EVALUATED until actual audited GPU execution. No model download, remote
access, Slurm submission or Git push is performed by this implementation.

The six delivery files are this document, `jobs/r5_i5r2c4_scores.sh`,
`scripts/141_r5_i5r2c4.py`, `src/semcache/edgelora/r5_i5r2c4.py`,
`src/semcache/edgelora/r5_i5r2c4_runtime.py` and `tests/test_r5_i5r2c4.py`.
Include all six in the operator-reviewed commit; preparing a real plan rejects
uncommitted/untracked execution sources. No local historical GPU evidence was
available, so the historical numerical counts above remain verified handoff
context rather than newly reproduced workstation measurements.
