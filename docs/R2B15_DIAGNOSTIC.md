# R2-B1.5 — numeric first divergence and warm cost isolation

Initial repository: `exp/raw-semcache-r2`, HEAD
`445b4d85775ee3e1063095f17b779e6053bef550`. Both R2-B1.5 specification documents
were read and govern this implementation. Their untracked files are preserved.
The B1 runner/runtime/cache/audit/tests/docs, earlier R1/R2-A code and results,
and Slurm scripts remain unchanged. Seven new implementation/test/doc files are
added. Classification: `EXACT_PREFIX_BASELINE_DIAGNOSTIC`,
`safe_semantic_reuse_claimed=false`.

Local status is **IMPLEMENTED_LOCAL_ONLY**. Torch, Transformers, PEFT and pytest
are unavailable here. Historical B1 strict FP16 failures and approximately 75 ms
warm observations are motivation only; no historical samples enter new results.
No actual GPU first-divergence, clone-cost or speedup finding is claimed.

## Reused contracts and functional gate

The new runner calls the existing `PrefixRuntime` and uses its OPT loader,
controlled PEFT fixture, exact token namespace, immutable independent prefix
snapshot, mandatory per-request copies, actual installed cache-API detection,
position/mask audit and fixed `atol=rtol=1e-3` metrics. Default flags remain eager,
use_cache=True, output_attentions=True, return_dict=True, batch one, same adapter,
eval and inference mode. Downloads remain disabled by default.

For every phase, an untimed gate verifies native Full with L tokens and both
Normal/Prepared with S=L−P tokens at every decoder layer/QKV/FFN. It reuses B1's
actual input IDs, learned-position embedding and causal-mask checks, P→L cache
length and canonical source checks. Normal and Prepared logits must be exactly
equal and their physical row audits identical. Source derives only from B1's
independent P-token prefill. No Full-derived prefix KV is ever supplied to Warm.

A structural, namespace, cache-mutation or Normal/Prepared parity failure stops
the phase. FP16 Full-relative STRICT_FAIL alone does not stop diagnosis. Source
hashes and model-weight hashes are checked outside measured request timers.

## A — numeric observation semantics

`--phase numeric` performs the following untimed experiment:

1. Two uninstrumented Full calls check repeatability.
2. A third Full call captures selected checkpoints as a bounded CPU reference;
   its logits must exactly match the plain Full.
3. Another independent Prefix Source call captures and compares actual prefix
   checkpoints to Full `[0:P]`. Its K/V must match the canonical independent
   source used for Warm. This extra source pass is diagnostic work, not an
   amortized service/source cost.
4. Plain Warm establishes output parity. A hooked Warm compares its suffix
   checkpoints to Full `[P:L]` and must exactly match plain Warm logits.
5. Compare actual Full prefix K/V against Source K/V, and Source K/V against
   Warm's returned past. These are comparison-only slices; they never populate
   the serving cache.

Comparisons are explicitly separate:

| Comparison | Reference | Candidate | Meaning |
|---|---|---|---|
| FULL_PREFIX_vs_SOURCE | Full token rows 0:P | independent Source rows 0:P | source/full shape-conditional drift before reuse |
| FULL_SUFFIX_vs_WARM | Full rows P:L | Warm rows 0:S | includes source KV, suffix shape and accumulated differences |
| FULL_PREFIX_vs_SOURCE_KV | actual Full cache prefix | actual Source cache | actual post-head-split causal KV, not raw QKV projections |
| SOURCE_vs_WARM_PAST | Source KV | returned Warm cache prefix | preserved past; no source replacement |

For attention probabilities, reference axes are `[batch,head,query,key]`:
Prefix comparison slices both query and key to P; Suffix comparison slices query
P:L and keeps all L keys. FFN's native flattened `[tokens,features]` input is
normalized to `[1,tokens,features]` for comparison, while original native shape
and stride remain recorded. Shapes/dtypes must match after explicit slicing;
broadcasting is rejected. KV metric coordinates flatten heads/features to
`[batch,token,head*head_dim]`, with native shape/stride/head_dim reported separately.

Observed modules include token/learned-position embeddings, layer input/output,
attention and FFN LayerNorm inputs/outputs, QKV inputs/Base/final PEFT outputs,
actual scaled LoRA B outputs, post-softmax attention probabilities, weighted-V
context entering out_proj, attention output projection, attention module output,
FC1/activation/FC2, decoder final LN and LM Head input/output. Module or callable
activation is observed through a scoped hook/wrapper calling the original exactly
once and returning the original result. All hooks/wrappers are removed in finally.

Attention probabilities are observed at attention-module return, after the
output projection has already executed. Their values describe post-softmax
probabilities, but their hook timestamp is not the softmax execution point.
Consequently first-observed context/output drift cannot establish that those
arithmetic operations preceded probability drift.

Raw pre-softmax scores and inline residual additions lack stable module boundaries
and remain **unsupported**, rather than being independently reimplemented.
Residual effects are bracketed by layer/LN/FFN observations. CUDA kernel identity
is not collected. Checkpoint absence is disclosed, missing required hooks fail.

`execution_order` is actual hook order, not sorted layer number. The summary
separates:

- `first_bitwise_difference`: actual byte-representation difference, including
  signed zero; first element and absolute token position are recorded.
- `first_nonzero`: value inequality under torch.equal, which treats +0/−0 equally.
- `first_tolerance_failure`: fixed torch.isclose/allclose threshold violation.

These are first **observable checkpoints**, not the first underlying instruction
or an identified cuBLAS kernel. Sparse `--layers 0,1,2,4,8,16,last` limits the scope
accordingly. Default `all` examines every layer. Full references live on CPU and
are cleared even on failure. Default reference budget is 1024 MiB; over-budget
runs fail explicitly and can be rerun into a new directory with fewer layers.
One staging tensor and numeric-metric scratch memory are additional to this
retained-reference budget. No long-lived all-layer GPU activation snapshots or
activation/attention/KV dumps are written to disk.

### Narrow same-input shape probe

`--shape-probe auto` triggers only after an observed prefix projection difference.
It replays one targeted Full to retain the actual Full hidden input for that layer,
checks that input against the CPU Full reference, then uses the same native PEFT
module/weights on Full L rows and an exact prefix selection of P rows. Base,
scaled LoRA delta and final output are compared separately using existing C1
`capture_call`. Actual operand shape/stride/dtype/device are recorded. Only one
layer's live GPU hidden input is retained and released promptly.

The probe is `NON_SERVING_DIAGNOSTIC`; its times are not request measurements.
Observed fixed-input shape sensitivity may be SUPPORTED/CONTRADICTED in this
fixture. The probe also reports whether actual Source projection input was exact;
if an upstream LN/input already differed, this is not a unique attribution of the
first Source drift. No Full-derived cache control or precision remediation is
implemented. Unique initial arithmetic/kernel cause remains UNRESOLVED unless
supported by additional causal evidence.

## B — four cost modes and exact exclusions

| Mode | Implementation | Clock/interpretation |
|---|---|---|
| FULL_USE_CACHE | unchanged B1 native Full | minimal synchronized request wall; L logits/output KV |
| WARM_NORMAL | unchanged B1 EXACT_PREFIX_WARM | input/positions, namespace, clones, factory, native Suffix and output KV inside original request wall |
| WARM_PREPARED_DIAGNOSTIC | same B1 warm call; instance materialize hands off a fresh prepared cache | **IDEAL_DIAGNOSTIC**: clone/factory work excluded; input/positions, namespace/shape guards and native Forward retained |
| PREPARE_ONLY | original snapshot.materialize without model Forward | independent component measurement; never a request/speedup sample |

Prepared uses a **new single-use ticket for every request**. Its actual K/V copies
and any DynamicCache conversion complete before the request timer starts. Heavy
content checks, actual independent source equality, exact target/snapshot/namespace,
position, dtype/device and pointer/version guards are verified outside timing.
At the original B1 materialize call site, the handoff retains namespace/token/mask
and cache shape checks and adds a small single-use/binding/version check. Only
that snapshot instance method is temporarily replaced; Normal and old B1 code
are unchanged. The handoff must occur exactly once. A mutated/stale ticket is
invalidated; a consumed or failed ticket cannot serve another request. Method
restoration occurs on exceptions. No pool of reusable prebuilt mutable caches.

Prepared's result records the cache's before-P/after-L state and excluded
preparation wall/bytes. The original B1 result still declares its normal clone
bytes; B1.5 explicitly labels that inherited field and separately records actual
in-request explicit clone bytes (Normal only). Container internals can copy or
allocate again, and their exact allocation bytes are not invented.

### Timing boundary audit

The original B1 timer ends after native model output and CUDA synchronization,
before cache-content checks, CPU logits transfer and Python frame cleanup. This
is preserved. Native output-cache construction is inside the model forward.
Isolated post-return cleanup time is unavailable and recorded **null/unsupported**;
no claim is made that it was measured inside the original timer. The enclosing
invocation wall in breakdown also includes wrapper setup, audits/export and,
for Prepared, prebuilding/checking the excluded ticket. It is not minimal latency.

`--phase timing` uses no profiler, added module hooks or detailed event book.
Warmup >=3, FP16 measured >=10 (one exploratory run). Source/canonical build and
heavy content/hash checks stay outside. Full/Normal/Prepared order uses seeded
Latin blocks; incomplete final blocks are disclosed by the saved schedule.
Full computes L logits while Warm computes S; this baseline output difference is
reported rather than hidden. No independent-run CI or B2 speedup conclusion.

`D(F,H)=Full−Normal`, positive favors Normal. Raw rows also contain
Normal−Prepared and Full−Prepared, with mean/p50/p95/std, wins and descriptive
Full-relative ratios. Normal−Prepared can include request-entry cache, allocator
and host scheduling effects; it is not a clone-only causal estimate. Prepared
cannot support a service-speedup claim. `correctness_qualified_speedup=false`
throughout this diagnostic stage.

### Instrumented breakdown

`--phase breakdown` is a separate pass with uninstrumented warmups followed by
1–3 instrumented repeats and PREPARE_ONLY. Scoped wrappers use CPU wall and CUDA
events without repeated component synchronizations; resolve event intervals only
after the enclosing request/component synchronization. Actual model values and
outputs must match uninstrumented Warm. Event insertion changes scheduling, so
these wall times cannot enter minimal summaries.

Recorded boundaries and nesting:

- adapter selection/eval and input-mask-position preparation: request children;
  tokenization is outside because input is pretokenized.
- cache_materialize: inclusive request child containing namespace, actual clone
  loop, optional factory and shape checks.
- namespace_guard, kv_clone, cache_container: children of materialize; never add
  their values to their inclusive parent. Clone call count/explicit byte count
  reflect actual helper invocations; factory may copy additional bytes.
- native_forward: includes native attention/FFN/LM Head/output-cache work.
- lm_head: nested inside native_forward; never add to the parent.
- return_cleanup: unsupported independent boundary; null.

CUDA stream intervals can include host launch gaps and are not GPU kernel busy
time. Host values are inclusive function scopes, not profiler CPU self time.
CPU/GPU/parent/child clocks are never simply summed. Kernel launches, actual
memcpy event count and allocator-operation bytes remain null without a profiler;
Torch allocated/reserved/peak memory is recorded separately. Profiling traces and
output_attentions=False ablations are not implemented in this narrow stage.
GPU monitoring is block-boundary only; unavailable UUID/clock/process evidence is
`MONITORING_UNAVAILABLE` and does not itself fail inference.

## Artifacts and failure behavior

Use new `results/r2/b15/<tag>/` directories; collisions fail, no resume/overwrite.

- `manifest.json`: actual Git/code/spec hashes, packages/installed B1 API/flags,
  phase, identity, source costs, preflight, seed and schedules.
- `numeric_first_divergence.csv`: chronological scalar observations, both branches,
  native/logical shapes/strides/dtypes and bitwise/value/strict metrics.
- `numeric_summary.json`: first findings, baseline quality, unsupported boundaries,
  shape-probe/evidence statuses, CPU-reference size and causal limitations.
- `cache_kv_parity.csv`: actual layerwise prefix K/V comparisons.
- `projection_shape_probe.csv`: conditional same-input Base/LoRA/total controls.
- `timing_paired_samples.csv`, `timing_summary.csv`: minimal-only request samples,
  orders, paired differences/ratios, tickets and quality.
- `component_boundaries.csv`: instrumented scopes, parent relationships, clocks,
  explicit bytes/calls, memory and null unsupported metrics.
- `cache_integrity.json`: B1 physical/token/mask/position audit, Normal/Prepared
  exact parity, canonical checks and final model-weight hashes.
- `gpu_state.json`, `decision_summary.json`, and `failure.json` on failures.

Unused phase artifacts retain empty schemas/UNVERIFIED states. Numeric candidate
rows are checkpointed between Source/Warm passes even if a late probe fails.
A strict FP16 mismatch can coexist with COMPLETE execution. Missing evidence,
mutation, instrumentation parity failure or API/position errors cannot be promoted
to physical/accuracy success. Unknown component residuals are not forced to sum
to a historical or minimal request latency.

## SERAPH commands

No Slurm files are created. Use allocated GPU resources and the existing offline
environment; downloads require explicit opt-in. Pin model/tokenizer revisions
when reproducing a particular B1 snapshot. These commands match runner86 help.

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# FP32 Smoke: numeric + 3-repeat minimal + one-repeat breakdown, separate tables.
python scripts/86_diagnose_r2b15_prefix.py --phase smoke \
  --model facebook/opt-125m --dtype float32 --cases 32:16,32:31 \
  --device cuda:0 --cache-device cuda:0 --warmup-runs 3 --measured-runs 3 \
  --output-dir results/r2/b15/smoke125m_01

# FP16 numerical diagnosis. Reduce layers in a NEW tag if CPU reference budget fails.
python scripts/86_diagnose_r2b15_prefix.py --phase numeric \
  --model facebook/opt-2.7b --dtype float16 --cases 128:64,128:127 \
  --device cuda:0 --cache-device cuda:0 --layers all \
  --reference-limit-mib 1024 --shape-probe auto \
  --output-dir results/r2/b15/numeric27b_01

# Minimal Full/Normal/Prepared paired comparison; no detailed hooks/profiler.
python scripts/86_diagnose_r2b15_prefix.py --phase timing \
  --model facebook/opt-2.7b --dtype float16 --cases 128:64,128:127 \
  --device cuda:0 --cache-device cuda:0 --warmup-runs 3 --measured-runs 10 \
  --order-seed 701 --output-dir results/r2/b15/timing27b_01

# Separate component pass, including PREPARE_ONLY; these are not minimal samples.
python scripts/86_diagnose_r2b15_prefix.py --phase breakdown \
  --model facebook/opt-2.7b --dtype float16 --cases 128:64,128:127 \
  --device cuda:0 --cache-device cuda:0 --warmup-runs 3 --instrumented-repeats 1 \
  --output-dir results/r2/b15/breakdown27b_01
```

Only the listed two-case workload families are accepted. P=0 control remains in
frozen runner85. No larger context sweep, full-derived KV leakage control,
precision remediation, output flag ablation, new policy or compression is added.

## R2-B2 gate and current uncertainty

Proceed to B2 only with actual installed API/physical-skip/source integrity and
Normal/Prepared exact parity evidence. Preserve and report strict FP16 failures;
Prepared's idealized result is not a substitute for Normal service latency.
B2 must keep the same output/cache/backend/precision endpoint, charge all Normal
copies/preparation, distinguish source construction/amortization and residency,
and collect independent runs before making statistical speedup claims. R1/R2-A
use_cache=False numbers cannot be pooled with this use_cache=True baseline.

Local tests validate scalar alignment, chronology, prepared ownership/mutation,
clock separation, artifact collisions and failures. Real tiny random OPT/PEFT and
CPU/CUDA tensor tests are provided but require dependencies. They do not download
models or stand in for OPT-2.7B GPU evidence. Full local regression commands:

```bash
PYTHONPATH=src:tests python3 -m unittest test_r2b15_numeric test_r2b15_cost \
  test_r2b1_prefix_cache test_r2b1_prefix_runtime test_r2a_overhead_audit \
  test_r2a_runtime test_r2a_profiler test_r1b3_gemm_shape \
  test_r1b3_b3_correctness test_r1b3_b3_break_even test_r1b3_b21_mapping \
  test_r1b3_batched_executor test_r1b3_fast_selection test_r1b3_occurrences \
  test_r1b2 test_r1b1_d1 test_r1b1_impact test_m9a2_impact_audit \
  test_m85_alignment test_r1_raw_semcache -q
```

Local verification record: **17 new tests collected, 12 passed, 5 skipped**.
The listed combined R1/R2-A/B1/B1.5 regressions collected **208 tests: 147 passed,
61 skipped, zero failures**. Syntax checks and CLI help passed. Pytest is absent;
these were unittest runs. Fake event/cache/control-flow tests are not GPU evidence.

The actual local Smoke CLI failed at the Torch import before any model Forward.
`results/r2/b15/local_validation_20261028/` preserves its required empty schemas,
failure/traceback, UNVERIFIED physical/correctness status, and seven implementation
SHA256 values under IMPLEMENTED_LOCAL_ONLY. No GPU phase or profiler was run.
Model-dependent tests and all actual FP16 observations remain unverified.
