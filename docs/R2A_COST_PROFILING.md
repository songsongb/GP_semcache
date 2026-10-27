# R2-A: RAW QKV cost profiling (2026-10-27)

This is a diagnostic harness, not an executor optimization. Classification remains
`ORACLE_DIAGNOSTIC`, `safe_reuse_claimed=false`. FP16 tolerance remains
`atol=rtol=1e-3`. No prefix KV reuse, compression, network simulation or new policy.
The supplied R1 reference (Full 76.09/77.62 ms; Contiguous 140.66/180.83 ms at
128/256 tokens) is **historical**, not a result produced by this implementation.

## Repository and execution contracts

Implementation started on `exp/raw-semcache-r1b3`, HEAD
`c48450ca8cbe7461e22b49f613afb0226ea964f2`. The user-supplied untracked
`docs/R2-A_실험명세_2026-10-27.md` and its Zone.Identifier were preserved.
Each run records its actual branch, HEAD, status, SHA256 of all local Python
sources/scripts, package versions, resolved model/tokenizer revision, adapter
parameter hashes, source/token identity, GPU UUID and driver inventory. Branch
names and historical fixture reproduction are never assumed.

Inspected execution lineage: scripts 69/72 load eager OPT + controlled PEFT and
use `metrics/r1_runtime.native_request` / `engine_request`; 75 supplies source
cloning and Fast selection; 76/77 supply executor choice and RequestMapping;
79/80 use `diagnostics/r1b3_b3_runtime.Runtime`; 81 aggregates independently of
forward execution; 82 supplies actual hidden-input capture and correctness-only
shape controls. R2-A calls these components without editing them.

| Mode | Actual execution | Omitted operations |
|---|---|---|
| FULL_NATIVE | Existing R1 `native_request`: tokenize, adapter/eval, native full PEFT model | All SemCache control/policy and reuse |
| LOOKUP_NO_REUSE | Existing B3 `request(mode='lookup')`: semantic encoding, cluster update, exact-w3 lookup, legacy nonoverlap selection, native-only Legacy wrapper, attention impact and policy | Fast oracle selector, physical reuse and mapping |
| CONTIGUOUS_WARM | Existing B3 `request(mode='contiguous')`: full Engine frontend/policy, Fast selector, mapping cache lookup, validation, Contiguous executor | Production eligibility for the oracle source catalog |
| ORACLE_PRESELECTED | Tokenize + adapter/eval + same model call inside unchanged `batched_v2_projection_path(contiguous=True)` with prepared real source hits/mapping | Semantic/clustering, lookup, selection, mapping-cache lookup/build, attention-impact, admission/CHU/PBR, Engine logging/control |

Oracle **retains** executor entry snapshot validation, module validation, patch
setup/cleanup, fresh native PEFT computation, all non-QKV model work and actual
cached QKV assembly. It is not a hypothetical zero-validation executor. Its
preselection/mapping cost is separately recorded. `CONTIGUOUS − ORACLE` removes
an explicit bundle including *post-forward policy*, not lookup alone.
No-reuse executes the original candidate selection, but its execution hits are
empty. It is not the Fast-selector oracle with reuse disabled.

All modes share exact token IDs, `user_a`, model weights, eager attention,
`output_attentions=True`, `use_cache=False`, batch one and prefill-only semantics.
The endpoint is prompt-to-logits, **not generated TTFT**. Source data comes only
from actual source prefill. The source/backing values are hashed and checked;
no Target FULL tensors populate the catalog. Source hashing is correctness-only
and increases the separately reported source construction cost.

## Measurement scopes

### Minimal

No Torch profiler or added module hooks. Existing R1 synchronized request wall
is used. Native/Oracle CPU logits transfers and correctness checks occur after
the timer. Oracle wall includes tokenization, adapter/eval, input preparation,
executor setup/validation, model and cleanup. Engine modes include real frontend,
impact and policy in their original timer. The reused R1 harness clones Engine
state before timing and validates/exports audit after timing; those operations
are not silently redefined as R1 request wall. Their broader invocation cost can
appear in traces and is explicitly outside the reported minimal request scope.

Source creation/model initialization are outside target timing. Warm mapping is
prepared explicitly before the block; its actual hit status is checked in every
measured Contiguous request. Oracle has `PRESELECTED_NOT_A_CACHE_LOOKUP` rather
than inventing a 100% cache-hit rate. Warmup defaults to 3, pilot measured 10,
smoke measured 3. Seeded cyclic Latin blocks balance mode positions over complete
blocks of four; ten repeats have an incomplete final block. All raw samples and
order are retained. GPU monitoring and output I/O occur outside request timers.

`D(X,Y)=T_X−T_Y`; positive means X slower. The four same-session bridges are
Contiguous−Full, Lookup−Full, Contiguous−Oracle, Oracle−Full. Bootstrap CIs are
within-run paired exploratory intervals, not three-independent-run evidence.
No outlier removal. Strict FAIL remains visible even when execution completes.

### Profile trace

`--phase trace` first collects a fresh minimal block, then profiles 1–3 requests
per mode. Smoke also does this. Profiler wall samples are never included in the
minimal CSV/summary. Supported `torch.profiler.profile`, CPU/CUDA activities,
`record_function`, record_shapes, profile_memory and Chrome export are detected
at runtime. Unsupported CUDA Kineto/profiler fails the requested phase explicitly;
minimal rows already collected survive. Stack capture is optional.

Ranges surround actual operations, with scoped wrappers removed on exit:

- Embedding, LayerNorm, Base Q/K/V, actual LoRA A/B children, attention module,
  attention output projection, FFN up/down, LM head.
- Semantic encode, cluster assignment, window extraction, cache lookup, Fast
  selection, index/snapshot validation and mapping-cache operations.
- Existing Engine timing scopes, token-precompute impact, and unchanged
  executor's source resolution/conversion/gather/scatter/fresh/allocation spans.

Parent QKV hooks are **not installed**; the executor forbids them. Python
interception, FFN activation, LoRA cast/scaling and unfused attention internals
can remain in an enclosing range or unknown. No fake zero-cost attribution.
Attention's parent range is inclusive in the trace; CPU operator table uses
**self** time only, categorized by closest actual range.

`profiler_operator_cost.csv` reports CPU self time, device self-attributed time
(if exposed by the installed Torch), calls, operand shapes and self memory.
`profiler_gpu_kernel_cost.csv` separately aggregates raw kernel/runtime/memcpy
activities by name/category/component/stream. Explicit External id or correlation
links connect activities to actual range ancestry. Unlinked kernels remain
unknown; names are never used to infer cuBLAS algorithms. Activity counts are
observed trace counts, not inferred PyTorch operation counts. Missing memcpy
byte counts stay null. Raw trace retains individual event timestamps.

GPU activity **union** and duration **sum** are both recorded. Cross-stream
overlap prevents treating the sum as elapsed time. CPU self, device-attributed
operators, raw kernel activities and request wall must never be summed together.
QKV raw kernel duration/fraction and known-attribution fraction are evidence,
not end-to-end savings. Trace covers the broader Runtime invocation, including
post-request logit transfer/audit; it is not identical to the R1 timer boundary.
`profiler_request_ms − same-session minimal mean` is a descriptive sequential
phase overhead estimate, not a paired causal estimate. Profiler memory accounting
is allocation activity, not live resident bytes; allocator peak is separate.

One trace/request bounds retention. Default export budget is 64 MiB per trace;
oversize export is removed from the new run only and the phase fails with
`TRACE_BUDGET_EXCEEDED`. This is an artifact limit, not a hard bound on profiler
CPU RAM before export. Existing CUDA memory preflight is an estimate, not an OOM
guarantee. No all-layer tensor dumps or long traces are produced.

### Components

Layer-0 hidden states retain their actual shape, stride, dtype/device. Full and
fresh-subset use the same actual model weights. Native PEFT total, Base, and
actual LoRA cast/dropout/A/B/scaling are timed independently. Cached contiguous
slice/conversion/copy uses the actual source backing. Snapshot validation and
warm mapping-cache lookup are separate host/component measurements. Each has
3 warmups followed by configurable repetitions. CUDA events, if present, measure
stream intervals including launch gaps; they are **not kernel busy time**.
Components use deliberate synchronization between repetitions. They do not
represent full-model latency, total layer scaling, or an optimized executor.
Microbench input preparation occurs outside the projection component timer and
is recorded as such. These rows are never subtracted from whole-model wall.

## Correctness and physical audit

After minimal timing, Full/Full repeatability, Full-relative strict/argmax/KL
metrics and Legacy/Contiguous/Oracle exact QKV comparisons run untimed. Reused,
fresh and assembled values, final logits, selected source occurrence sequence,
and actual native/skipped row counters must agree for reuse executor parity.
Native Full projection shapes are observed in a separate untimed pass. No-reuse
must have zero physical skipped rows. Actual source hashes must match backing.
Missing evidence, parity failure, preflight/OOM, unsupported requested profiler
and runtime exceptions remain failures; they cannot become COMPLETE. Partial
CSV and detailed failure traceback/context remain available.

`execution_status=COMPLETE` can coexist with `correctness_status=STRICT_FAIL`.
No correctness-qualified speedup is inferred from argmax agreement or row skips.

## Artifacts

Required files are created even if model import fails:

- `manifest.json`: identity, semantics, settings, status, source costs, workload,
  source/backing hashes, memory screen, schedules and raw-only guard evidence.
- `minimal_paired_samples.csv`: phase/repeat/order/mode, synchronized wall,
  actual coverage/row audit, mapping hit, memory and strict metrics.
- `minimal_summary.csv`: measured-only mean/p50/p95/std per length/mode.
- `overhead_bridge.csv`: paired differences, CI, wins with explicit sign.
- `profiler_operator_cost.csv`, `profiler_gpu_kernel_cost.csv`,
  `profiler_trace/*.json`: disjoint reporting views of instrumented execution.
- `component_microbench.csv`: isolated real-tensor component measurements.
- `correctness_and_coverage.json`: Full repeatability, exact executor parity,
  backing identity, physical storage and source costs.
- `decision_summary.json`: H-A…H-E judgments, profile evidence/overhead and
  unknown residuals; `gpu_state.json`: block-boundary monitoring.
- `failure.json` on failure: type, context and traceback.

Unused phase CSVs have headers and no measurements. New output directories are
mandatory; collisions fail. There is no resume or aggregation in this initial
pilot harness. A failed run must be rerun into a different tag; partial pairs
must not enter summaries. No historical rows are imported.

## Interpretation and next-stage gate

H-D/H-E use the relevant paired CI sign: entirely positive supports slower
Lookup/Oracle respectively, entirely negative contradicts, crossing zero is
unresolved. This is descriptive one-session evidence, limited to the implemented
Oracle's retained validation/setup. H-A/H-B/H-C remain UNRESOLVED until attributed
trace and component evidence are reviewed; their raw evidence is exported and
no automatic causal verdict is forced from overlapping timings.

- H-A: inspect Full QKV vs all kernel duration, explicit linkage completeness,
  unknown cost, and busy union vs wall. Skipped rows are not full-model FLOPs.
- H-B: inspect semantic/control/policy and assembly CPU self/launch ranges against
  Contiguous−Oracle. This contrast cannot alone distinguish host from policy GPU.
- H-C: inspect full/subset Base/LoRA component cost, actual operand shapes,
  launch and memory activities; do not infer efficiency or kernel choice by name.
- Prefer **R2-D investigation** if controllably removing host/executor work closes
  much of the Full gap; this is not permission to omit necessary service work.
- Consider **R2-B exact-prefix Forward reuse baseline** if even this QKV-only
  Oracle remains slower and the recoverable QKV fraction is small. R2-A does not
  implement prefix reuse or establish its safety.

## SERAPH CLI

All examples use new directories; no Slurm files are generated. Offline is the
default loader behavior; model download requires explicit `--allow-download`.
Use the same resolved revision flags/seed as a reference if exact reproduction
is required. The runtime records resolved revisions even if flags are omitted.

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/83_profile_r2a_costs.py --phase smoke \
  --model facebook/opt-125m --dtype float32 --prompt-lengths 32 \
  --device cuda:0 --cache-device cuda --warmup-runs 3 --measured-runs 3 \
  --output-dir results/r2/a/smoke125m_01

python scripts/83_profile_r2a_costs.py --phase minimal \
  --model facebook/opt-2.7b --dtype float16 --prompt-lengths 128,256 \
  --device cuda:0 --cache-device cuda --warmup-runs 3 --measured-runs 10 \
  --order-seed 101 --output-dir results/r2/a/pilot27b_01

python scripts/83_profile_r2a_costs.py --phase trace \
  --model facebook/opt-2.7b --dtype float16 --prompt-lengths 128,256 \
  --device cuda:0 --cache-device cuda --warmup-runs 3 --measured-runs 10 \
  --trace-repeats 1 --max-trace-mib 64 \
  --output-dir results/r2/a/trace27b_01

python scripts/83_profile_r2a_costs.py --phase microbench \
  --model facebook/opt-2.7b --dtype float16 --prompt-lengths 128,256 \
  --device cuda:0 --cache-device cuda --microbench-repeats 10 \
  --output-dir results/r2/a/components27b_01
```

Trace/microbench runs include their own unprofiled paired anchor. For another
independent pilot, change both `--independent-run-id`, `--order-seed`, and output
tag. Do not merge profiler latencies into minimal samples.

## Local verification

The local environment lacks Torch, Transformers and PEFT. CLI help, syntax,
scalar pairing/artifact/failure tests run without them. Torch-backed executor
and CPU/CUDA profiler tests are explicitly skipped; fake-event tests verify
accounting logic, not real GPU behavior. No new R2 GPU latency or root-cause
finding is claimed. Local verification: 161 collected tests, 112 passed, 49 skipped,
zero failures (including R1 regressions). R2-A targeted: 15 collected, 12 passed,
3 skipped. Syntax and CLI help passed. The actual local smoke CLI stopped at
`No module named 'torch'`; `/tmp/r2a_local_smoke_20261027` preserves its
`FAILED / UNVERIFIED` manifest and required empty schemas. This is not a model
smoke PASS.

```bash
PYTHONPATH=src:tests python3 -m unittest test_r2a_runtime test_r2a_profiler -v
PYTHONPATH=src:tests python3 -m unittest test_r1b3_gemm_shape \
  test_r1b3_b3_correctness test_r1b3_b3_break_even test_r1b3_b21_mapping \
  test_r1b3_batched_executor test_r1b3_fast_selection test_r1b3_occurrences \
  test_r1b2 test_r1b1_d1 test_r1b1_impact test_r1_raw_semcache -q
```

API reference: [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html).
Runtime feature checks take precedence over documentation for the installed
SERAPH version. Occupancy, Tensor Core utilization and specific cuBLAS algorithm
identification require additional evidence such as Nsight; none is claimed here.
