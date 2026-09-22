# M9-A.2 Semantic Impact overhead audit

This is an opt-in research diagnostic. The production reducer, M7/M8/M8.5 engine,
admission/eviction rules and existing M9-A results are unchanged. No benchmark
numbers are supplied until the audit runs on SERAPH. The approximately 664 ms
reference is user-reported, not a locally reproduced measurement.

## Trace and structural findings

`SemCacheEngine.query` starts `attention_impact_timer`, iterates **all candidate
windows**, calls `actual_attention_impact` -> `PaperRowL2SumReducer.reduce`, groups
repeated cache keys by their mean impact, and appends a query-history observation.
The measured scope includes grouping/history but excludes subsequent CHU/policy.

For each window the production reducer iterates every layer and:

1. Calls `attention.detach().double()` on the **entire** attention tensor, then
   slices query rows. The slice is a view; FP16-to-FP64 conversion allocates a new
   tensor. `masked_fill` allocates another tensor for the selected rows.
2. Rebuilds causal validity, copies the engine's CPU validity mask to the device,
   masks the selected rows and computes finiteness.
3. Converts the finiteness scalar to a Python bool, an implicit CUDA host wait.
4. Computes row L2, averages heads, sums the window, then `.item()` materializes
   the scalar on the host, another implicit CUDA wait.

At 30 windows × 32 layers, code execution therefore entails 960 layer visits,
full attention conversions, row-norm/head-reduction calls, mask-to-device calls,
and masked copies, plus 960 boolean and 960 impact-scalar host reads. Blocking
CPU-mask-to-CUDA copies add waiting beyond those scalar reads. The production
reducer has no explicit `cuda.synchronize()` call; implicit waits still matter.
The engine creates 30 CPU masks. Overlapping w=3 windows repeatedly reduce the
same token rows: 90 query rows per layer versus 32 for precomputation.

These are structural causes of repeated work, not measured attribution of the
664 ms. Python dispatch, kernel launch, FP64 conversion/arithmetic, allocator and
host waits must be distinguished using the audit data. No claim is made that any
single operation explains the observed latency or that speedup will be 30×.

## Formula and implementations

All modes implement the unchanged formula:

```
r[t] = sum_l mean_h sqrt(sum_{valid causal k} A[l,h,t,k]^2)
I([i,j)) = sum_{t=i}^{j-1} r[t]
```

The layer/token row-L2 sum is `PAPER_DEFINED`; head mean and interpreting t as a
query row with all valid causal keys are `REPRODUCTION_CHOICE`. Actual full-prefill
attention values are used, without block renormalization. Input attention dtype
and values are preserved. **All three retain the existing reducer's explicit
FP64 arithmetic**, even for FP16 input; this is reported as `reduction_dtype`.

- `CURRENT_BLOCKWISE`: instrumented copy of the production operation order.
  A separate uninstrumented pass invokes the actual production reducer. Both
  are checked against values captured from the actual engine invocation.
- `TOKEN_PRECOMPUTE`: compute r once, build a device-side window-membership mask,
  and reduce each window's selected r values in a batch.
- `TOKEN_PREFIX_SUM`: compute the same r, prefix-sum it, and index P[end]-P[start].
  A fixed-order doubling scan uses deterministic device-side elementwise operations
  (five steps at length 32); it does not disable the fixture’s determinism setting
  or require backend support for deterministic floating-point `cumsum`.

The alternatives loop over layers, not candidate windows, for GPU reductions.
One packed host transfer includes all impacts and a finite flag. Causal/padding
masks are retained; nonfinite values outside selected valid query/key cells are
ignored just as in production. No-candidate input produces no reductions.

Floating-point summation order differs in the alternatives. Each value must pass
`abs(error) <= 1e-10 + 1e-12 * abs(reference)`; max absolute and relative L2 errors
are recorded. Policy decisions require **exact equality**, even if impacts pass
tolerance. No threshold rounding or score clamping hides differences.

## Measurement and attribution

The driver loads local assets only, at recorded model/tokenizer snapshots, and
recreates the controlled PEFT cold -> same_user_exact fixture. It checks the
source environment hash, adapter initialization metadata, host/GPU identity,
token IDs, reuse spans/count and existing output-parity tolerances. The source
must be correctness-valid length-32 M8.5 plus strict base-only ES attachments.
Capture retains the real physical-reuse attention tensors on the original device;
base-only attention is not substituted. Model/cache/encoder remain resident.

The benchmark reuses those fixed tensors for 3 discarded warmup rounds and 10
measured rounds per implementation, rotating implementation order. This isolates
impact work; it does **not** measure 10 new inference requests or decode. Timings
include repeated-key aggregation and history append on an independent copy of the
captured history. Cold-query reductions and policy checks are outside target timing.

Every raw row records dimensions, per-layer attention shapes, heads, device/dtype,
observed operation counts, synchronization locations, impacts and:

| Field | Wall-clock scope |
| --- | --- |
| `impact_attention_access_ms` | Validation, indexing, FP64 conversion dispatch; optimized mask/index preparation |
| `impact_row_l2_reduction_ms` | Causal masking, validity, norm/head reduction dispatch; current mask copy |
| `impact_device_to_host_ms` | Scalar bool/item or packed `.cpu().tolist()`; includes queued GPU completion |
| `impact_window_aggregation_ms` | Window sums or device prefix/gather |
| `impact_finalize_ms` | Repeated-key means, history append, and separately reported unscoped host overhead |
| `impact_total_ms` | Entire instrumented impact scope |

The five wall components sum to `impact_total_ms`.
`impact_unscoped_host_overhead_ms` is included in finalize, not another additive
component. `*_device_ms` CUDA event spans are additional overlapping diagnostics,
not summable with wall time and not pure isolated kernel time (launch gaps and
copies can occur inside them). D2H wall time is **not** pure PCIe transfer time.

Synchronization is explicit in the artifacts:

- A pre-measurement `cuda.synchronize` excludes outstanding forward/previous work.
- Current bool/item reads synchronize per block/layer; blocking mask H2D calls
  are counted separately.
- Optimized code does one result materialization; mask/index host-to-device
  construction still incurs waits, and is counted explicitly.
- The final attribution event is synchronized outside total to resolve CUDA
  event spans. CUDA-event record counts are reported.

Instrumentation can perturb thousands of small operations. Therefore every round
also measures `uninstrumented_impact_total_ms`. Both speedups and the difference
between the two passes are reported. Only the **mean uninstrumented total** is
used for system recomposition. The optimized uninstrumented path retains audit
Python bookkeeping but disables stage clocks/events; the current uninstrumented
path calls production code directly. This is conservative and explicitly diagnostic.

## Policy and system gates

Tensor-free policy replay invokes existing `GlobalCache`, normalization,
admission/eviction, physical-hit selection, metric and CHU implementations.
Replay is first checked against actual captured engine admission/eviction and
selection events. Each implementation is then checked for identical decisions,
resident keys, selections and score parity, including scores when no eviction
occurs. At two queries PBR does not run and is not claimed as exercised. Tests use
a constrained cache to exercise real eviction. This is fixture-specific evidence,
not a proof for all threshold/tie cases or all workloads.

The existing strict `m9a_summary.json` and its companion `m9a_environment.json`
provide calibrated UD values and exact tensor/wire byte choices. Their source ES
hash, base profiles, control components and request identities must match the
supplied strict input. PEFT proxy and invalid reuse inputs are rejected. Diagnostic
results do not overwrite any source file; hashes are rechecked on completion.

For each original source repetition and implementation:

```
new_control = original_control - original_attention_impact + measured_new_impact
Edge(R) = original_ES + original_UD + original_edge_bytes * .008 / R_Mbps
SemCache(R) = new_control + original_remaining_ES + original_remaining_UD
             + original_remaining_bytes * .008 / R_Mbps
delta(R) = Edge(R) - SemCache(R)
```

All other terms stay fixed. Rows are produced for 200/500/1000 Mbps. If
`D = original_ES + original_UD - new_control - remaining_ES - remaining_UD`, then
`delta(R) = D + saved_bytes*.008/R_Mbps`. When D<0 and saved_bytes>0, the analytical
break-even is `saved_bytes*.008/(-D)` Mbps. Otherwise the output explicitly reports
all-bandwidth reuse, recomputation, or tie; no infinity is written as JSON.

Even the remeasured current baseline is a diagnostic recomposition. Every modeled
row is `RESEARCH_EXTENSION_DIAGNOSTIC` / `SIMULATED_RESEARCH_EXTENSION`. Optimized
impact measurements are `MEASURED_RESEARCH_EXTENSION`; current measurements are
`MEASURED`. They are not paper SemCache results or measured end-to-end latency.
If value/decision parity fails, raw evidence is saved, the run exits unsuccessfully,
and no recompositions are emitted. `--impact-only` explicitly omits this stage.

## Outputs and commands

New output directory contents:

- `audit_manifest.json`, `fixture_capture.json`: identities, input hashes, hardware,
  production capture checks and policy events; no serialized GPU tensors.
- `impact_raw.jsonl`: warmup/measured timings, counts, impacts, errors, decisions.
- `impact_summary.json` and `.csv`: measured mean/p50/p95 for all timing fields,
  both speedups, errors and policy parity.
- `system_diagnostic.json` and `.csv`: individual source-repeat recompositions.
- `system_summary.csv`: per-mode/per-bandwidth mean cost and analytical break-even.

From the repository root on SERAPH, with the previously documented artifacts:

```bash
# OPT-125M smoke: source artifact specifies model/dtype/revisions; no system costs.
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 scripts/36_audit_m9a2_semantic_impact.py \
  --es-input results/m9a1/opt125m_strict_smoke_tokenizer_fix/strict_es_input.jsonl \
  --impact-only --device cuda --warmup-runs 1 --measured-runs 2 \
  --output-dir results/m9a2/opt125m_impact_smoke

# OPT-2.7B FP16 length-32 full impact audit + separate system diagnostic.
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 scripts/36_audit_m9a2_semantic_impact.py \
  --es-input results/m9a1/opt27b_len32_strict/strict_es_input.jsonl \
  --system-summary results/m9a1/opt27b_len32_strict/costs/m9a_summary.json \
  --device cuda --warmup-runs 3 --measured-runs 10 \
  --output-dir results/m9a2/opt27b_len32_impact_audit
```

These commands consume existing artifacts and cached model/encoder/tokenizer
assets. Adjust input paths if SERAPH used different directory names; use a new
output directory on reruns. There is no download flag. Both model identity and
FP16/length-32 constraints come from the validated strict input, not CLI guesses.
No new CPU calibration or ES base profile is run by the audit.

Model-free regression, including synthetic tensor tests where torch is available:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_m9a2*.py' -v
```

Local validation for this change: 10 new model-free tests passed; 8 synthetic
Torch/CUDA tests skipped because Torch is unavailable. Existing M9-A/M9-A.1:
25 passed; M8.5: 9 passed, 2 tensor skips; tokenizer provenance: 12 passed.
CLI help, syntax compilation and diff checks passed. The command-driver tests
use mocked capture/benchmark functions and temporary files, never models.
No models, downloads, real timings or real system results were produced in Codex.
