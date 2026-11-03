# R5-I5R-2: historical numerical fidelity diagnosis

This delivery implements the CPU-only historical-evidence stage. The operator's
latest instruction is to analyze the existing SERAPH evidence **before proposing
new GPU inference**. Accordingly, no GPU capture mode, diagnostic serving change,
new GPU Slurm script or operation replay experiment is introduced here. Those
steps are deferred until the historical report identifies an actual evidence gap.
No original I5, R0 or R1 output/manifest is changed. No push, remote deployment,
Slurm submission, GPU inference or model download was executed.

## Status and verified versus reported facts

Historical raw evidence is on SERAPH, not this local workstation:

* Plan: results/r5/i5r/controlled_hit/plan_smoke_v1
* Run parent: results/r5/i5r/controlled_hit/run_smoke_attempt_01
* Recorded source: de06fcc6634323a78826b218901a821179991c26

The real raw tensors and R1 GPU audit exports are not present locally. Local
results are synthetic tensor/schema tests only, not newly reproduced GPU findings.
The operator previously verified one real 64/256-token physical HIT, eight equal
generated tokens, numerical tolerance failures on all eight steps, worst max-abs
0.03125 and relative-L2 approximately 0.002176. The frozen limits remain 0.01 and
0.001 respectively, alongside all existing elementwise/KL/argmax requirements.
The reported last-prefill-position trace showed equal layer-0 Q/K/V, the first
visible attention difference at layer 0 (0.001953125), and a layer-2 Q diagnostic
tolerance failure (0.0078125, 21 violations). These are a verified handoff, not
results computed locally by this patch, and not causal conclusions.

## Code inspected and reused

The production path is unchanged:

* r3b_runtime.physical_path calls the actual bare GPU Linear, obtains CPU LoRA
  correction through ProjectionSession, adds base plus delta, and casts back to
  base dtype. With FP16 base and FP32 delta this includes dtype promotion followed
  by FP16 rounding. It captures owned total raw Q/K/V before attention transforms.
* user_device.cpu_deltas casts hidden values to adapter matrix dtype, then runs
  F.linear(x,A), F.linear(intermediate,B), scale 1.0, and stacks Q/K/V. Prepared
  synthetic matrices are FP32, rank 8; actual precision receipts are retained
  when recorded. The ES copies the transported correction to its GPU.
* I5Owner admits completed standalone Raw-QKV with source/ownership/proof seals.
  Raw Q/K/V shape is [1,P,H]; attention K/V is [1,heads,P,H/heads].
* r4b2_kv.attention_kv performs a view, head/sequence transpose, contiguous copy
  and clone. It applies no K/V scaling or rotary transform. Public cache
  construction is checked for unchanged contents. Request-owned K/V copies
  protect resident storage from subsequent cache append/mutation.
* QualityDriver executes only target[prefix:] on a HIT, with past_length=prefix,
  complete causal mask and absolute positions. The installed eager OPT forward
  performs attention/cache append; the driver verifies the old prefix and newly
  appended Raw-KV slice, then continues normal autoregressive decode.
* r5_i5_audit independently checks the original TCP/ownership/consumed-KV receipts.
  Historical STRICT_AUDIT saves all-token raw QKV, initial returned KV, attention/
  FFN/layer outputs and cached raw snapshots. Decode QKV/components are saved;
  decode returned KV is generally hash-only. The old R1 layer summary reads only
  the last prefill position, which is insufficient for prefix equivalence.

Existing pure byte head transformation and frozen check_parity are reused.
The new analyzer does not import Torch, load a model or execute a forward. It
does not invoke or relax r5_i5_analysis.analyze's strict current-source check.

## New files

* src/semcache/edgelora/r5_i5r2_tensors.py — safe bounded input reader, semantic
  token alignment, chunked numerical metrics and first-observed partial order.
* src/semcache/edgelora/r5_i5r2_analysis.py — historical chain verification,
  prefix/source/cache/suffix/decode comparisons, missing-coverage and hypotheses.
* scripts/129_analyze_r5_i5r2.py — CPU-only CLI.
* tests/test_r5_i5r2.py — synthetic arrays/metadata, corruption and provenance tests.
* docs/R5_I5R2_NUMERICAL_FIDELITY.md — this operator and scientific handoff.

No existing production module, threshold or eligibility code is modified.

## Historical integrity and provenance contract

Select one original block, either explicitly with --run-dir or through the
single-block --runs-root selector. Original block names must match
block_<index>_<numeric-job-id>; sibling _audit directories are excluded. Duplicate
attempts/multiple original blocks are ambiguous and require explicit selection.
Missing/incomplete required evidence fails nonzero. There is no destructive
extraction, input rewriting, cache restoration or manifest resealing.

Every raw file is streamed through SHA-256 against the exact sealed inventory.
The original experiment's inventory hash, experiment/block identities, A/B/C
request order, immutable matched request hashes, model/adapter/tokenizer/GPU
identities, source ordering/sharing, token/position/mask and prefill/decode
contexts are checked. All tensor references are tied to the verified inventory,
including captures not selected for numerical comparison. Corrupt bytes, missing
required files, unsafe/symlink paths or mismatched shapes/dtypes fail closed.

The historical I5 exported audit is required and independently inventory-checked;
its configuration must match the raw configuration and its frozen thresholds
must match the unchanged numerical implementation. The R1 export is additionally
checked when present (or explicitly selected with --r1-audit-dir), retaining its
old last-position diagnostic without treating it as an all-token trace.

All recorded historical source-file hashes are checked against Git blobs at
--expected-source-commit, **not** against the new checkout's src/**/*.py inventory.
An available blob mismatch always fails. Missing historical objects also fail by
default. --allow-unavailable-historical-git is an explicit partial-provenance
option for portable copies: it reports HISTORICAL_OBJECT_UNAVAILABLE and
PARTIAL_HISTORICAL_PROVENANCE, never source PASS. It cannot waive a mismatch.
No object is downloaded. The current diagnostic commit/branch/file hashes and
tracked status are recorded separately.

This is HISTORICAL_EXPORT_AUDIT_PLUS_NEW_CPU_TENSOR_COMPARISONS. It is not a fresh
original-source GPU independent audit. If such an audit is separately needed,
use the clean recorded historical checkout/worktree and its original analyzer;
do not disable its source inventory check in the new checkout.

## Comparisons and interpretation

Default coverage is every recorded layer and every aligned token, not sampling.
--layers explicitly reduces layer coverage and is recorded in gate_layer_scope.
The analyzer computes:

1. Full-target prefix Q/K/V versus standalone C-source Q/K/V, all prefix positions.
2. Source raw Q/K/V versus physically cached raw Q/K/V.
3. Source K/V versus a byte-level head transform of retrieved cached raw K/V.
4. Source K/V versus consumed-input hash/layout receipts, separately labeled
   HASH_ONLY; no consumed-input array is invented from a hash.
5. Source K/V versus the prefix preserved in the HIT's returned prefill cache.
6. Full-target prefix K/V versus standalone source and HIT prefix K/V.
7. Full-target suffix Q/K/V, attention, FFN and layer outputs versus HIT suffix.
8. Full-target prefix attention/FFN/layer outputs versus the source's matching
   prefix positions, plus optional matching forensic internals if actually saved.
9. A/B full QKV and A/C standalone-source controls.
   Also compare short-source versus full-target-prefix QKV/components *within A*,
   isolating the length comparison from condition-to-condition differences.
10. Decode QKV/components only for equal prior generated histories, with actual
    absolute positions; prefill/decode logits for all matched requests, including
    priming, using the original frozen check_parity implementation.

Mask, position, token and generated-history alignment precedes comparison. A
standalone-source last position is never compared with an unrelated target last
position. Stored array indices and absolute semantic positions are separate.
Head-axis prefix slicing covers every head. Comparisons include shape, dtype,
device, byte identities, max/mean absolute error, relative L2, elementwise
violations, first differing position and full aligned-element coverage. Original
strides are NOT_OBSERVABLE_FROM_AVAILABLE_EVIDENCE when the old capture omitted
them; serialized shape does not prove the original tensor stride.

Activation checks use the frozen atol/rtol/max-abs/relative-L2 values but are
diagnostic, not a substitute for logit KL/argmax gates. Only explicitly marked
masked-score captures may contain negative infinity; NaN/unexpected infinity
is rejected. No float cast changes serving outputs or retroactively replaces FP16.

Independent statuses: CACHE_SOURCE_INTEGRITY, CACHE_RECONSTRUCTION_INTEGRITY,
FULL_PREFIX_KV_EQUIVALENCE, ATTENTION_INPUT_EQUIVALENCE,
ATTENTION_OUTPUT_EQUIVALENCE, PREFILL_LOGIT_FIDELITY, DECODE_LOGIT_FIDELITY,
TOKEN_TEXT_FIDELITY, FROZEN_NUMERICAL_TOLERANCE and NATIVE_PEFT_STRICT_PARITY.
Unobserved evidence is explicitly unknown; native PEFT remains UNESTABLISHED.
Numerical FAIL is a preserved scientific result, not a CLI integrity error.
CACHE_RECONSTRUCTION_INTEGRITY is specifically a recorded-value/layout/hash
verdict. A PASS does not validate an unrecorded original stride or actual kernel
mask, nor eliminate a layout-dependent attention-kernel effect.

FIRST_OBSERVED_DIFFERENCE is reported separately for source/full prefix,
full/HIT suffix, cache round trip and decode. Parallel Q/K/V operations have the
same causal rank. FIRST_CAUSALLY_LOCALIZED_DIFFERENCE and ROOT_CAUSE remain
UNRESOLVED until a genuinely discriminating controlled comparison supports them.
Layer-0 attention is not automatically the cause, and using FP16 is not proof of
FP16 shape dependence. H1 includes baseline controls; H2 bounds observed round
trip errors; H3/H4 remain unresolved without actual operands; H5 reports a
layerwise trajectory without assuming the causal chain.

## Bounds and outputs

The default tensor limit is 32 MiB (configurable up to 128 MiB); raw/audit trees
are each bounded at 32 GiB (hard maximum 128 GiB). JSON files are bounded at
16 MiB and JSONL files at 128 MiB, with 16-MiB line limits. Input inventories are
streamed; float metrics use chunks of 65,536 values. Two bounded tensors and
selected slices are held at a time. No model weights are loaded. A too-large
file produces an explicit bound failure; it is not silently sampled or skipped.

Fresh outputs only under results/r5/i5r/numerical_fidelity/ in the repository:

* i5r2_evidence_inventory.json
* i5r2_tensor_comparisons.json
* i5r2_first_divergence.json
* i5r2_hypothesis_matrix.json
* i5r2_numerical_fidelity.json
* i5r2_source_provenance.json
* i5r2_runtime_contracts.json
* i5r2_cpu_correction_receipts.json
* i5r2_historical_r1_export.json
* i5r2_root_cause_report.md
* i5r2_output_manifest.json

CPU precision/TCP hash receipts are preserved separately from actual numerical
tensor values. Effective GPU matmul/backend flags not recorded historically
remain unknown even if source code contains an intended setting. Performance
is deliberately excluded; neither this analysis nor future heavily instrumented
runs can be compared to old latency numbers as an equivalent benchmark scope.

## Commands: local Ubuntu

No extra packages are required for the analyzer: Python 3.10+ standard library
and local Git suffice. Do not run the real command against an invented local
path; the real SERAPH tensors are unavailable here.

```bash
PYTHONPATH=src:tests python3 -m unittest test_r5_i5r2 -v
python3 scripts/129_analyze_r5_i5r2.py --help
```

Final CPU validation: **23/23 new tests PASS**. The complete applicable suite
ran **228 tests: 224 PASS, 0 FAIL, 4 SKIP**. Skips are the existing pinned ROUGE
test, installed OPT contract, Torch tensor-ownership test, and actual HF OPT
cache/suffix test because their local dependencies are absent. They are not
reported as installed-runtime or GPU PASS. Exact combined command:

```bash
PYTHONPATH=src:tests python3 -m unittest \
  test_r5_i5r2 test_r5_i5r1 test_r5_i5r0 test_r5_i5 test_r5_i5_controller \
  test_r5_i5_preflight test_r5_i5_hotfix test_r5_i5_gpu test_r5_contracts \
  test_r5_semantic test_r5_controller test_r5_multi test_r5_fidelity \
  test_r5_perf_evidence test_r4c_opt_contract test_r3b_cache test_r4a_cache \
  test_r4b1_cache test_r4b2_cache -v
```

The synthetic CLI ran twice and produced byte-identical sealed artifacts under
results/r5/i5r/numerical_fidelity/cpu_validation_final_20261103/. Those deliberately
injected arrays demonstrate cache-integrity PASS alongside full-prefix numerical
FAIL; they are not measured model output. The test also confirms that input
bytes/manifests remain unchanged. CLI help, compilation and whitespace checks
passed. No process/TCP or GPU execution was necessary for this read-only patch.

Review and commit the five files above manually, then handle push/pull yourself.
No remote Git action is performed by the script. New src files changing the I5
collector are expected; this is why historical Git blobs and exports are used.

## Phase A: SERAPH historical review first, no GPU allocation

After the operator-managed reviewed source update:

```bash
cd /data/khuss/repos/GP_semcache
conda activate semcache
git cat-file -e 'de06fcc6634323a78826b218901a821179991c26^{commit}'
python scripts/129_analyze_r5_i5r2.py \
  --runs-root results/r5/i5r/controlled_hit/run_smoke_attempt_01 \
  --experiment results/r5/i5r/controlled_hit/plan_smoke_v1 \
  --expected-source-commit de06fcc6634323a78826b218901a821179991c26 \
  --output-dir results/r5/i5r/numerical_fidelity/historical_review_01
```

The parent must contain exactly one original block. If multiple attempts exist,
explicitly select one real block path with --run-dir; do not silently combine
retries. The default _audit and _i5r1_audit siblings are auto-detected. If exported
audits were relocated, pass --audit-dir and --r1-audit-dir with their actual paths.
All outputs must use fresh directories. Verify i5r2_source_provenance.json before
interpreting numerical observations. Missing old Git objects must be repaired
through an operator-reviewed historical checkout/object, or reported explicitly
as partial provenance; the script does not fetch or replace them.

An optional early-layer-only second review (reuses existing files; no inference):

```bash
python scripts/129_analyze_r5_i5r2.py \
  --runs-root results/r5/i5r/controlled_hit/run_smoke_attempt_01 \
  --experiment results/r5/i5r/controlled_hit/plan_smoke_v1 \
  --expected-source-commit de06fcc6634323a78826b218901a821179991c26 \
  --layers 0 1 2 \
  --output-dir results/r5/i5r/numerical_fidelity/early_layers_review_01
```

No R5 GPU opt-in, CUDA allocation, TinyBERT load or Slurm submission is needed.
Read-only hashing/CPU numerical analysis may take time; follow the university's
CPU allocation policy if login-node analysis is restricted.

## Phase B/C: deliberately deferred pending the real Phase A report

Return the source-provenance status, first-divergence report, hypothesis matrix
and exact missing comparisons before selecting new GPU work. The first question
is whether prefix Q/K/V already differs *before* admission. If round trip is
exact, distinguish that from full-prefix equivalence instead of repeating a
cache-integrity experiment. If equivalent raw operands exist but attention
differs, determine which actual scaled-Q/mask/score boundary is missing. If
projection differences already exist, identify whether hidden input, native
Linear output or CPU delta is unrecorded.

Only those demonstrated gaps should determine a future bounded early-layer
capture or isolated same-input operation test. Such work must retain the existing
RTX 3090/CPU/disk/source preflight, genuine standalone source, normal controller,
separate diagnostic-only scope and unchanged thresholds. No oracle full-target
prefix can become an authorized production cache. A future FP32 operation replay
cannot replace the failed historical FP16 verdict. No Phase B/C sbatch command
is supplied before the historical evidence has been analyzed, as requested.
