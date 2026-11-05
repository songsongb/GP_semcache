# R5-I5R-2C-3-B0: actual missing operand recapture

DIAGNOSTIC_ONLY. GPU_NOT_RUN locally. C2 job152894 observations are the operator's
verified handoff, not a new local reproduction. Original C2 and C3 Stage A output,
SHA-256 inventories, old Git objects and numerical thresholds remain immutable.

B0 reuses the frozen I5 `setup`, `QualityDriver.c_forward`, `physical_path`, actual
CPU UD `serve/cpu_deltas`, routed TCP, C2 `gate/padded`, installed OPT contract,
compatible-RTX3090 verification, writer accounting, tensor readers and seals.
One bare model is loaded once and one independent Adapter_A CPU UD is started.
The thirty-user/three-adapter registry is validated, but only User_01/Adapter_A
executes. No semantic encoder, cache admission, cache HIT or live cache restoration
is used in this operand recapture. It is not a fresh physical-reuse validation.

## Exactly what executes

Two full-model FP16 prefills execute, S64 then F256, using the exact original
SOURCE_ONLY Qasper tokens. S appends zero rows only to native Q/K/V Linear input
(64→256), at all32 layers, and keeps only real64 output rows; CPU LoRA still sees
64 rows. F is unchanged native Full Recompute with256 real rows. Masks, positions,
other hidden states and returned KV are never padded. Generated in-request past
is discarded after each prefill and is never supplied to F.

The frozen TCP completion protocol requires a genuine selected token to finish
each parent. B0 therefore changes only the diagnostic output budget to one,
selects the actual greedy token from each prefill's logits, and sends ordinary
COMPLETE_REQUEST controls. There are **two protocol-selected tokens, zero decode
forwards**, no generation loop, no text/quality benchmark and no fake completion
IDs. Full-vocabulary logits are computed by the existing forward but not dumped.
Budget changes are recorded; exact original token/mask/position/precision semantics
are checked and numerical reproduction must still pass. Source policy is already
one token; F's historical two-token budget is explicitly capped at one.

Passive hooks observe actual Layer0 total Q/K/V projection returns, complete
native additive mask and first64 module-output rows. Full S64/F256 Q/K/V and
masks are saved, with original shape/stride/offset/dtype/device and SHA-256.
No CPU/reference tensor equation supplies a GPU operand. Ordinary all-layer
forward evidence remains hash-only; only ten Layer0 tensor binaries are saved.
All32-layer CPU/TCP and source-padding receipts are audited without all-layer dumps.

No warmup or unrelated adapter forwards are added. The absence of the original
C2 warmups, changed allocation/layout/scheduling and extra capture may affect
reproduction; they are not grounds to waive the exact gate or automatically retry.
One RTX3090, four cores,64 GiB RAM,30-minute Slurm allocation,900-second program
bound,256 MiB saved-evidence/accounting bound,192 data-file bound. Diagnostic
timing is not production performance evidence. Failed directories remain intact.

## Independent reproduction gate

The CPU analyzer revalidates all seals, both source snapshots against their Git
blobs, the historical C2 exported audit, current base/adapter/tokenizer preparation,
actual CPU UD lifecycle and no-CUDA isolation, matching ES/UD frame hashes,
all32 real-row correction shapes and arithmetic precision receipts, source-only
zero padding/slicing and the current compatible GPU receipt. Runtime identities
are constructed with `runtime_view`; a base identity is not compared directly
with an adapter-extended identity. New and historical physical namespaces must
match except their explicitly recorded GPU UUIDs; no historical UUID is forged.

It checks complete causal masks, real position-module observations, new S/F
prefix equality, **each** new Q/K/V prefix against its corresponding old C2
prefix, **each** new module-output prefix against its corresponding old C2 output,
and source/full-mask first64 query rows against recorded old masks. Agreement
between new S and F alone is insufficient. Changed base, adapter, backend,
corrupt/missing tensors, incomplete forward/UD completion or invalid masks fail
closed. Numerical reproduction mismatches are sealed and exit2, preserving the
new result but forbidding C3 companion preparation.

Verdicts are independent: NEW_CAPTURE_COMPLETE/B0_CAPTURE_COMPLETE,
HISTORICAL_C2_PREFIX_REPRODUCED, HISTORICAL_C2_ATTENTION_REPRODUCED,
HISTORICAL_C2_MASK_REPRODUCED, COMPANION_REPLAY_ELIGIBLE/C3_COMPANION_READY,
HISTORICAL_REPLAY_NOT_ESTABLISHED. FIRST_ATTENTION_DIVERGENCE and
STRICT_NUMERICAL_FIDELITY_RESTORED are NOT_EVALUATED; ROOT_CAUSE_CONFIRMED is false.

## SERAPH: CPU preparation

Manually review/commit/push/pull all changed files first. B0 requires the same
offline semcache environment and original Git objects; nothing is downloaded.
Paths below are prompted because complete job152894 paths/source SHA are not
available locally. Select the original SOURCE_ONLY **mode** directories.

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
export I5_CONDA_ROOT="$(conda info --base)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
read -r -e -p 'Original C2 SOURCE_ONLY plan: ' C2_PLAN
read -r -e -p 'Original job152894 SOURCE_ONLY raw run: ' C2_RUN
read -r -e -p 'Corresponding exported C2 independent audit: ' C2_AUDIT
read -r -p 'Verified historical C2 source commit: ' C2_COMMIT
read -r -e -p 'Validated multi-adapter preparation directory: ' B0_MULTI
read -r -p 'Reviewed current B0 source commit: ' B0_SOURCE_COMMIT
export B0_SOURCE_COMMIT
test "$(git rev-parse HEAD)" = "$B0_SOURCE_COMMIT"
export B0_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export B0_ROOT="$PWD/results/r5/i5r/numerical_fidelity/attention_operand_capture"
export B0_EXPERIMENT="$B0_ROOT/plan_$B0_TAG"
export B0_RUN_ROOT="$B0_ROOT/attempt_$B0_TAG"
CUDA_VISIBLE_DEVICES='' python scripts/140_r5_i5r2c3b0.py prepare \
 --c2-experiment "$C2_PLAN" --c2-run "$C2_RUN" --c2-audit "$C2_AUDIT" \
 --expected-historical-commit "$C2_COMMIT" --multi-prepared "$B0_MULTI" \
 --output-dir "$B0_EXPERIMENT"
```

Historical model/adapter/backend/source verification precedes GPU allocation.
Set workspace exactly as recorded by validated C2 backend receipts; no unrecorded
backend alternative is allowed:

```bash
read -r -p 'Recorded CUBLAS_WORKSPACE_CONFIG (UNSET for null): ' B0_WORKSPACE
if [[ "$B0_WORKSPACE" == UNSET ]]; then
  unset CUBLAS_WORKSPACE_CONFIG
else
  export CUBLAS_WORKSPACE_CONFIG="$B0_WORKSPACE"
fi
read -r -p 'Permitted node (moana-y3 or moana-y4): ' B0_NODE
export B0_NODE B0_ENABLE_GPU=1
mkdir -p logs
B0_JOB_ID="$(sbatch --parsable --export=ALL --nodelist="$B0_NODE" jobs/r5_i5r2c3b0_operands.sh)"
export B0_JOB_ID
```

One manual job only; no automatic replay job is submitted.

## Independent CPU audit and C3 companion preparation

```bash
export B0_RUN="$B0_RUN_ROOT/run_${B0_JOB_ID%%;*}"
export B0_AUDIT="${B0_RUN}_audit_01"
CUDA_VISIBLE_DEVICES='' python scripts/140_r5_i5r2c3b0.py analyze \
 --experiment "$B0_EXPERIMENT" --run-dir "$B0_RUN" --output-dir "$B0_AUDIT"
```

Stop on a nonzero exit or any false reproduction/eligibility gate. If and only if
all required gates pass, prepare a separate C3 plan with explicit companion paths:

```bash
read -r -e -p 'Verified offline base_prepare directory: ' C3_BASE_PREPARED
export C3_EXPERIMENT="$PWD/results/r5/i5r/numerical_fidelity/attention_localization/companion_plan_$B0_TAG"
CUDA_VISIBLE_DEVICES='' python scripts/139_r5_i5r2c3.py prepare \
 --experiment "$C2_PLAN" --run-dir "$C2_RUN" --audit-dir "$C2_AUDIT" \
 --expected-historical-commit "$C2_COMMIT" --base-prepared "$C3_BASE_PREPARED" \
 --companion-experiment "$B0_EXPERIMENT" --companion-run "$B0_RUN" \
 --companion-audit "$B0_AUDIT" --output-dir "$C3_EXPERIMENT"
```

C3 independently recomputes the B0 gate and binds the sealed B0 audit to the
original C2 anchor. New operands are labeled
`NEW_B0_RECAPTURE_VERIFIED_AGAINST_C2`; original C2 module-output references remain
separate and authoritative. Missing C2 values are **not** relabeled recovered.
Original C3 preparation without companion flags still rejects incomplete original
evidence. The original Stage A verdict is never edited. Running C3 GPU replay is
a separate operator decision, using its existing wrapper and exact-replay gate.

## Files, local validation and limitations

Raw outputs: `i5r2c3b0_plan.json`, `i5r2c3b0_status.json`,
`i5r2c3b0_records.json`, `i5r2c3b0_backend.json`, `gpu_identity.json`,
`padding.jsonl`, ES/UD ledgers, lifecycle receipts, ten `tensors/*.bin`, and
`i5_output_manifest.json`. CPU audit: `i5r2c3b0_comparisons.json`,
`i5r2c3b0_decision.json`, `i5r2c3b0_independent_audit.json`,
`i5r2c3b0_report.md`, sealed inventory. Source/provenance and context are embedded
in plan, records and audit; original and current commits are kept separately.

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r2c3b0 -v
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r2c3 test_r5_i5r2c2 -v
bash -n jobs/r5_i5r2c3b0_operands.sh
python3 scripts/140_r5_i5r2c3b0.py --help
python3 scripts/139_r5_i5r2c3.py prepare --help
```

CPU fixtures and mocked export checks never establish genuine GPU recapture or
historical reproduction. Installed OPT hook tests require Torch/NumPy/Transformers
and are explicitly skipped when absent. No GPU jobs, remote deployment, Slurm
submission, historical overwrite or Git push is performed locally.

Final local validation: **49 PASS / 0 FAIL / 3 SKIP** across52 tests: B0 focused
19 PASS/1 SKIP; relevant C3/C2 regressions30 PASS/2 SKIP. The skips require absent
Torch/NumPy/Transformers APIs (native OPT hooks, attention observer, FP16 padding).
CLI help/required companion-path rejection, Python AST, shell syntax and diff
checks pass. A regression explicitly checks that passive projection hooks install
through the existing observer **after** `physical_path`'s bare-module guard.
Independent audit requires the same reviewed capture source snapshot.

Review and commit these eight files together; do not stage unrelated results/logs:

```bash
git add src/semcache/edgelora/r5_i5r2c3b0.py \
 src/semcache/edgelora/r5_i5r2c3b0_runtime.py \
 src/semcache/edgelora/r5_i5r2c3.py scripts/139_r5_i5r2c3.py \
 scripts/140_r5_i5r2c3b0.py jobs/r5_i5r2c3b0_operands.sh \
 tests/test_r5_i5r2c3b0.py docs/R5_I5R2C3B0_OPERAND_CAPTURE.md
git diff --cached --check
git diff --cached --stat
```

No commit or push is executed by Codex. The operator chooses the reviewed commit
and handles deployment and the single manual GPU submission.

## Contract rejection and failure-finalization hotfix

B0 and C2 construct contracts with the same sequence: `runtime_view`,
`QualityDriver.configure`/`installed_contract`, then `physical`'s adapter/registry
extension. Static inspection alone does not identify the observed differing field.
No field is normalized or ignored. The mismatch remains unresolved until actual
expected/current values are compared.

Before rejecting, B0 writes `i5r2c3b0_contract_difference.json`: exact RFC6901 JSON
paths, missing/extra/type/value classifications, canonical-JSON SHA-256 hashes,
types and bounded safe scalars. Source implementations and whole configuration
objects are never dumped in this receipt. Limits are64 differences and20,000
visited nodes, with explicit truncation; full-contract hashes and exact equality
are checked independently of report truncation. Early backend/allocation receipts
are saved without claiming any KV creation before a prefill has executed.

`seal` is explicitly imported. Cleanup, reconciliation, status writing and sealing
are attempted independently so secondary failures cannot replace the primary
exception (including on Python3.10). FAILED status retains the primary reason and
secondary stage errors. If the accounting writer fails, only the fresh attempt's
FAILED status may use a direct fallback write; accounting remains explicitly
FAILED, and no COMPLETE/audit gate is bypassed. Available evidence is sealed when
storage permits. A sealing failure is recorded as such and never claimed sealed.
Existing attempt directories are still refused; preserve failed attempts and
prepare a fresh sealed plan under the newly reviewed commit before retrying.

Hotfix validation: **53 PASS / 0 FAIL / 2 SKIP** in the B0/C3 focused suites
(55 tests), plus Python/shell syntax and diff checks. Skips are the unavailable
native Torch/Transformers CPU APIs. No GPU execution or historical evidence
mutation was performed, and no GPU contract equality has been newly verified.
