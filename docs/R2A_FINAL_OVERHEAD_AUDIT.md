# R2-A5 — profiling boundaries and controlled validation isolation

Repository inspected: `exp/raw-semcache-r2`, HEAD
`df9aecce8e043db34d5ccb7ab285f0eb0270d72b`. The user's untracked
`R2-A_OPT27B_결과분석_2026-10-27.md` is preserved. This stage adds only
script 84, `diagnostics/r2a_overhead_audit.py`, its tests, and this document.
Scripts 69–83, R1/R2-A runtimes, executors, selectors, policies, tolerances and
previous results remain frozen. No Slurm files or git push.

## Confirmed static findings

**Warm Mapping Lookup includes snapshot validation in the microbench.**
`r2a_runtime.microbench` calls `PreparedMappingCache.get` without a `validate`
argument, so `validate=True`. Its first substantive operation is
`descriptor.validate(hits,book)`. A warm cache hit does not bypass that check.
The historical 11.68/23.69 ms validation and 11.81/23.77 ms mapping means therefore
measure overlapping work in separate trials. They must not be added together.
Subtracting those independently timed means is also not a causal measurement of
the pure dictionary lookup. The new A/B directly removes one precisely identified
validation call inside the same Oracle request.

**Actual normal Contiguous requests use a different mapping boundary.**
`RequestMapping.select` calls `cache.get(...,validate=False)`; validation is
performed once by `batched_v2_projection_path` at executor entry. Fast selection
also calls `index.guard` before candidate matching, and B3 Runtime calls
`index.guard` after the request timer. These are separate scans, not additional
calls hidden in the normal warm mapping-cache lookup.

**`attention_inclusive` in the exported kernel table is a closest-tag category.**
The module hook surrounds `self_attn`, including its QKV projections, assembly,
attention mathematics and output projection. R2-A attribution selects the nearest
nested `r2a::` range. It therefore partitions tagged operations rather than
reporting an inclusive attention-parent kernel total. Reuse adds executor tags
inside that parent. LoRA cast/scaling and projection-side operations can also
move between residual parent and executor tags. This provides a code-supported
reason why the label's cost can decrease without skipping attention mathematics.
The exact historical kernel movement remains **UNRESOLVED** because original
Chrome traces are absent locally; the analysis markdown/aggregated values cannot
recover kernel ancestry or correlation links.

## Boundary table

All numeric timings below are unmeasured in this local A5 work. Historical means
are contextual evidence from the supplied R2-A report only.

| Component | Exact boundary | Relationship | Evidence / conclusion | Uncertainty |
|---|---|---|---|---|
| Warm mapping microbench | synchronize/events/start → `cache.get(validate=True)` → events/synchronize | Includes descriptor validation, key formation, cache lookup, prepared-index check, LRU and counters | **SUPPORTED:** overlaps standalone validation | Pure lookup incremental cost requires direct intervention |
| Snapshot validation | `SourceBackingDescriptors.validate` entry through catalog/source/owner/entry/bucket/view checks and return | Child of default get and executor setup | **SUPPORTED:** CPU structural/version/shape/pointer scan | CUDA-event interval includes host gaps; not proof of GPU kernel work |
| Normal warm mapping | `RequestMapping.select` timer around `get(validate=False)` | Separate from executor validation | **SUPPORTED:** microbench default differs from normal request | Parent setup/mapping timers must retain original definitions |
| Selection | `FastOccurrenceIndex.select`: layout/guard, windows, prefix, identities, candidates, nonoverlap, coverage, provenance | Contains index validation; B3 `selection_ms` closes **before** `flow.select`/mapping | **SUPPORTED:** selection is not lookup alone | Source index construction is separate |
| Attention impact | Engine `attention_impact_timer` after forward through precompute/window grouping/history append | Post-forward sibling of model time; profiler function range is narrower | **SUPPORTED:** Oracle omits this operation bundle | `compute_window_values` range omits outer Python grouping/history |
| Policy | CHU, admission/dedup/insert/materialization/events, and PBR timers | Materialization nested in admission policy; CHU/PBR separate | **SUPPORTED:** no independent sum of policy + materialization | Only triggered branches execute |
| Executor setup | Entry identity checks, then `setup_started`, validation, mask/audit, module patching until yield | Validation inside setup; initial checks before setup timestamp | **SUPPORTED:** mapping build and context setup cannot be blindly added | Broader request includes adapter/tokenization and all executor checks |
| Cleanup | `finally` deleting patched instance forwards | Separate after forward; included in Oracle request wall | **SUPPORTED** | Failure cleanup uses same finally |
| Fresh projection | `fresh_native_qkv`: `native(fresh_inputs)` | Includes Base/A/B children; indexing and allocation are siblings | **SUPPORTED:** physically fresh-only QKV | Inclusive native and child times overlap |
| Fresh assembly | `fresh_assembly`: `output.index_copy_(...,fresh)` | Child of mixed projection and attention parent | **SUPPORTED** | Does not include cached source scatter |
| Cached assembly | source resolution, cached gather/view, `.to(output)`, cached scatter/slice copy | Children of mixed projection; conversion can be no-op | **SUPPORTED** | Kernel/copy bytes require trace/audit, not `.to` call count |
| Attention category | Closest tagged operator/kernel, versus containing self_attn parent | Parent and nearest-child views overlap | **SUPPORTED:** reduced category cost does not establish attention skip | Actual historical per-kernel differences unresolved |
| Historical kernel movement | Requires original Full/Oracle Chrome CPU/GPU links | Cannot reconstruct from aggregate CSV | **UNRESOLVED** locally | Offline trace audit accepts the original files |
| Repeated validation causal cost | Validated Oracle minus prior-validated Oracle | Controlled request intervention | **UNRESOLVED** until GPU A/B | Standalone microbench means cannot substitute |
| Attention physically skipped | Frozen executor patches only Q/K/V instance forwards | No self_attn/FFN bypass implemented | **CONTRADICTED** as an explanation based solely on this reuse algorithm | Installed attention backend/kernel path still needs trace evidence |

`boundary_audit.csv` reproduces this schema with source path/symbol/line, source
SHA256, static status, uncertainty and null measured differences. A completed
paired run appends its actual controlled difference; static claims are not
converted into newly measured GPU facts.

## Controlled Oracle intervention

New mode `--phase paired --allow-validation-bypass` compares:

- `ORACLE_VALIDATED`: unchanged `CostRuntime.invoke('ORACLE_PRESELECTED')`.
- `ORACLE_PRIOR_VALIDATED`: the exact same invocation, with only the specific
  `runtime.plan.descriptor.validate` instance method temporarily replaced for its
  one executor-entry call. A minimal exact-hit/thread guard remains in this
  replacement; it returns no fabricated elapsed measurement.

Prior checks can warm CPU metadata and change request entry conditions relative
to historical R2-A runs. This A/B estimates the effect in its controlled warmed
fixture; it must not be subtracted directly from historical means.

Both conditions retain the same selected occurrences, source backing, mapping,
model, adapter, token IDs, projection modules, native fresh-token shape, assembly,
module safety checks, mask creation, patching, cleanup and request timing.
Existing Oracle/Contiguous/R1 modes are not changed. No class-level validation
setting is disabled. The instance method is restored on success and exception.
An AST contract check fails if the expected one executor validation call or
`get(validate=True)` default drifts.

The experiment obtains an exclusive **serial immutable-fixture lease**:

1. Full descriptor validation before starting the controlled case.
2. Exact metadata and tensor/version/pointer/shape/stride/device checks, exact
   selected sequence and mapping/index geometry, model parameter identity/version,
   adapter state and tokenizer roundtrip checks. Unexpected hooks, other Python
   threads, training mode or mismatch refuse the bypass.
3. Both conditions get one full prior and one full post validation outside their
   request timers. The bypass context owns these checks for its condition.
4. Actual reused/fresh/assembled QKV and logits are compared exactly in an untimed
   validation pair. Every measured pair again checks exact logits, selection and
   native/skipped row audit.
5. Source backing and mapping index **content hashes**, and all model-weight
   content hashes are checked at block boundaries. This catches persistent `.data`
   mutations that evade version counters. Hashing streams tensors to CPU without
   storing tensor dumps; for 2.7B FP16 the model hash reads approximately 5.4 GB at
   each boundary. This setup cost is outside target request time and not hidden as
   an execution optimization. The setup wall time and individual outside-request
   check costs are saved with their enclosing boundaries.

Hashes and full postchecks establish the controlled fixture after the block.
Until then samples remain `PENDING_POSTCHECK`. Mutation, missing/extra bypass
calls, missing QKV evidence or parity failure invalidate the case and leave
`UNVERIFIED`, partial samples, failure context and traceback. There is no claim
of protection against external writers or malicious mutation-and-restoration.
The lease is a restricted experiment, not a general caching safety mechanism.
Unsupported/stability-failing fixtures do not receive bypass timing conclusions.

All results remain `ORACLE_DIAGNOSTIC`, `safe_reuse_claimed=false`, explicitly
not production-safe. FP16 Full-relative strict failure is preserved at
`atol=rtol=1e-3`; `execution_status=COMPLETE` is distinct from `STRICT_FAIL`.

## CPU/GPU attribution audit

No new profiling forward is needed. `--trace <original Chrome JSON>` audits
existing traces offline without editing them or replacing frozen R2-A output.

| Attribution | Audit rule |
|---|---|
| QKV Base / LoRA | Actual Base/A/B child-module ranges and linked CPU operators; no kernel-name inference; cast/scaling may belong to parent |
| Attention | Report nearest tag separately from all kernels linked to operations within the self_attn parent; these views MUST NOT be summed |
| Attention mathematics | Explicit linked CPU schemas such as `aten::bmm`, softmax or SDPA inside attention are candidates; not a claim of skipped QK/V work |
| Impact | Separate post-forward range; not the self_attn parent and not the whole Engine impact timer |
| FFN | Actual fc1/fc2 ranges; activation/residual/normalization may remain outside them |
| Fresh native | Inclusive executor range containing Base/A/B operations; child totals overlap |
| Assembly | Actual fresh index_copy and cached gather/conversion/scatter ranges; do not attribute whole parent mixed time as assembly |
| Launch/copy | Explicit CUDA runtime/driver correlation and raw device activity; CPU launch wall is not GPU duration; conversion may be no-op |
| Unknown | Missing or conflicting External-id/correlation links stay ambiguous/unattributed |

The offline audit uses CPU-op origin IDs plus enclosing annotation containment,
tracks potentially conflicting launch links, and exports both old attribution and
new evidence. Parent/child kernel duration views intentionally overlap. It does
not sum them to end-to-end latency, infer cuBLAS algorithm/occupancy, or infer
attention semantics from kernel names. Identical source/input/executor outputs
alone cannot prove identical kernel timings or eliminate GPU clock variation.

## Running the narrow audit

Default `static` phase needs no Torch. It produces a source audit table; optional
traces are read only. Output directory collisions fail. There is no resume and no
broad profiler/sweep. Paired phase defaults to warmup 3 / measured 10 with seeded
balanced A/B ordering, all samples retained and within-run paired bootstrap CI.
Positive `validated − prior_validated` means the repeated validation increased
request latency. A CI entirely positive is `SUPPORTED`, entirely negative
`CONTRADICTED`, crossing zero `UNRESOLVED`; these labels concern this one fixture
intervention only. A request difference need not equal a standalone microbench
mean because of host/GPU overlap and execution state.

```bash
python scripts/84_audit_r2a_overhead.py --phase static \
  --output-dir results/r2/a5/local_boundary_audit_20261027

# Optional existing traces; no GPU profiler is enabled by this command.
python scripts/84_audit_r2a_overhead.py --phase static \
  --trace /path/to/FULL_NATIVE.json --trace /path/to/ORACLE_PRESELECTED.json \
  --output-dir results/r2/a5/existing_trace_audit_01
```

SERAPH environment:

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Start with only OPT-125M FP32 Smoke:

```bash
python scripts/84_audit_r2a_overhead.py --phase paired --allow-validation-bypass \
  --model facebook/opt-125m --dtype float32 --prompt-lengths 32 \
  --device cuda:0 --cache-device cuda --warmup-runs 3 --measured-runs 3 \
  --output-dir results/r2/a5/smoke125m_01
```

After fixture and QKV/output parity pass, run only the requested two lengths:

```bash
python scripts/84_audit_r2a_overhead.py --phase paired --allow-validation-bypass \
  --model facebook/opt-2.7b --dtype float16 --prompt-lengths 128,256 \
  --device cuda:0 --cache-device cuda --warmup-runs 3 --measured-runs 10 \
  --order-seed 501 --output-dir results/r2/a5/validation27b_01
```

Downloads default to disabled. Pin `--revision`, `--tokenizer-revision` and
`--encoder-revision` to reference resolved revisions when reproducing historical
fixtures. Actual revisions, branch/HEAD/status/source hashes, workload IDs and
order are recorded. Existing CUDA memory preflight runs; OOM is a failure.

Artifacts: `manifest.json`, `boundary_audit.csv`, `paired_samples.csv`,
`correctness_and_fixture.json`, `controlled_difference.json`,
`trace_attribution_audit.json`, `gpu_state.json`, `decision_summary.json`, and
`failure.json`/`summary.json` on failure. Static-only runs leave unavailable
measurements explicitly UNVERIFIED/null. Full tensors are never dumped.

## Final R2-A decision and R2-B requirements

The overlapping mapping/validation microbench boundary and non-inclusive meaning
of the closest-tag attention category are confirmed from code. The cost of
removing one repeated executor validation remains an empirical A5 question until
GPU A/B runs complete. Historical request gaps support continued investigation;
they do not independently identify all overheads or prove a safe speedup.

Proceed to an **R2-B exact-prefix KV baseline** that measures how much whole-prefix
Forward work is actually avoided, rather than extrapolating from QKV rows skipped.
It must verify identical model/adapter/token IDs/absolute positions and causal
prefix; strict logits against Full; source-build/amortization costs; cold/warm
KV residency and transfer; remaining suffix attention and FFN; paired unprofiled
Full-relative latency. Keep the semantic-window Oracle result and its FP16 Strict
FAIL separate from exact-prefix correctness claims. R2-A5 implements no R2-B path.

## Local verification (no model/GPU claim)

Local Torch/Transformers/PEFT remain unavailable. New tests: **13 collected,
10 passed, 3 skipped**. Combined R1/R2-A/A5 regressions: **174 collected,
122 passed, 52 skipped, zero failures**. Syntax and `--help` pass. Fake-fixture
and fake-trace tests check control flow/accounting only. Torch-backed tests cover
actual Oracle QKV/output parity, CPU/CUDA source execution and mutation refusal
when those dependencies are present; they were not executed locally.

The static CLI completed and saved its boundary table under
`results/r2/a5/local_boundary_audit_20261027_audited/` (ignored experiment outputs).
The real OPT-125M paired CLI stopped at `No module named 'torch'`; its manifest
is `FAILED`, fixture and correctness `UNVERIFIED`, preserved under
`/tmp/r2a5_local_smoke_20261027_final/`. This is not a GPU smoke pass. No original
Chrome traces were present locally, so historical kernel attribution is still
unverified; the report states that explicitly.

```bash
PYTHONPATH=src:tests python3 -m unittest test_r2a_overhead_audit \
  test_r2a_runtime test_r2a_profiler test_r1b3_gemm_shape \
  test_r1b3_b3_correctness test_r1b3_b3_break_even test_r1b3_b21_mapping \
  test_r1b3_batched_executor test_r1b3_fast_selection test_r1b3_occurrences \
  test_r1b2 test_r1b1_d1 test_r1b1_impact test_m9a2_impact_audit \
  test_m85_alignment test_r1_raw_semcache -q
```
