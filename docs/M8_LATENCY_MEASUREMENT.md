# M8 latency and memory measurement

M8 adds measurement around the validated M7 path; it does not change matching,
admission, CHU, PBR, eviction, or reuse authorization. Reuse remains exact,
cluster-scoped, non-overlapping `w=3` token matching. Every result sets
`safe_reuse_claimed=false`.

## Models and reproduction choices

`facebook/opt-125m` validates instrumentation cheaply (float32 by default).
`facebook/opt-2.7b` is the main RTX 3090 model (float16 by default).
`facebook/opt-6.7b` is an optional scale check only: it requires
`--allow-large-model`, is capped at two warmups and five repetitions, and is
excluded from tests/default runs. Dtype is always emitted, so 125M float32 and
2.7B float16 are not presented as a model-size-only comparison.

No revision is invented in source. An optional requested revision may be passed
on the CLI; `resolved_model_revision` and `resolved_tokenizer_revision` come
from the loaded Hugging Face objects and are written to results. Loading keeps
the established `use_safetensors=False` OPT compatibility behavior.

Default warmup/measured counts are 5/20 (125M), 3/10 (2.7B), and 2/5 (6.7B).
These are **REPRODUCTION_CHOICE**, not paper-defined. Every measured repetition
is retained. Warmup records are discarded, never folded into aggregates.

## Modes

- `NATIVE_NO_CACHE`: normal PEFT/OPT eager prefill; no semantic frontend,
  lookup, cache policy, or patched projections.
- `SEMCACHE_LOOKUP_NO_REUSE`: the M7 semantic, cluster, exact matching, lookup,
  and policy path runs, but the projection executor receives no reusable hits.
  It therefore projects every row and reports zero physical reuse/skipping.
- `SEMCACHE_PHYSICAL_REUSE`: the unchanged M7 selected hits enter the mixed QKV
  executor, which physically avoids native PEFT QKV calls for reused rows.

The fixture trace is recreated for every warmup/repetition and contains a cold
miss, exact same-user repeat, exact cross-user repeat, and unrelated query.
Reported reuse ratios are calculated from actual accepted non-overlapping spans;
no target ratio is fabricated. Exact parity is only labelled for controlled
identity cases and is not generalized to arbitrary cross-user reuse.

The state policy is `reconstructed_precondition`. Every SemCache trace starts
with a newly constructed clusterer initialized from the same anchor vectors, an
empty cache, and fresh F/A/I/history counters. Within that trace:

1. `cold_miss` begins from the empty cache and primes exact blocks.
2. `same_user_exact` begins from the deterministic state produced by the cold
   request.
3. `cross_user_exact` begins from the deterministic state produced by the cold
   and same-user requests, including their Eq.9/CHU changes.
4. `unrelated` begins from the same deterministic preceding trace.

Thus each aggregate condition has a logically equivalent starting state across
repeats. State reconstruction and prior-condition priming are outside the
condition's measured request interval. This is not an evolving state shared
between repetitions.

## Clocks, synchronization, and scopes

CPU control-plane scopes use `time.perf_counter_ns()`. CUDA execution uses
`torch.cuda.Event(enable_timing=True)`. Events are recorded on the current
stream and only the ending event is synchronized when its value is resolved.
There is no pre-region global synchronize. `request_wall_ms` includes the wait
needed to resolve its GPU work. Resetting/reading peak memory occurs outside the
measured request.

CUDA event creation/recording and host timing have nonzero overhead. In
particular, `mixed_qkv_execution_ms` records one event pair per Q/K/V projection,
but those child events never synchronize per layer or projection. Resolution
synchronizes the enclosing prefill end event once, then reads every completed
child pair without another synchronization. It is intended to locate cost, not
represent an uninstrumented production run. Warmups and distribution summaries
reduce, but do not remove, this effect.

Passive projection captures remain as detached tensors on the execution device
during the forward (without an additional clone);
there is no per-projection `.cpu()` transfer. For admitted misses, compact spans
move to the configured cache storage device later in `cache_materialization_ms`.
For hits in a CPU-resident cache, the required CPU-to-GPU QKV retrieval remains
real mixed-path work and is intentionally included in `mixed_qkv_execution_ms`.

For a measured CUDA request there are exactly two explicit synchronization
calls in the benchmark: one before capturing/resetting allocator statistics to
establish the memory baseline, and one on the enclosing prefill end event to
resolve final timing. The first is outside request latency. The second is the
unavoidable completion barrier for reporting elapsed GPU time and is included
in wall latency. Reading mixed-QKV child events adds zero synchronizations. Any
device transfer intrinsically required by CPU cache storage is execution work,
not an event-resolution synchronization call.

The attention-impact reducer likewise accumulates all layer/head norm tensors
before one host materialization per evaluated block; it no longer materializes
one result per Transformer layer. Multiple candidate blocks still require their
own impact reductions as genuine M7 control-plane work. `attention_impact_ms`
uses CPU wall time, not CUDA Events, because its scope deliberately includes the
host wait caused by each block's GPU-to-CPU norm-vector materialization.
`attention_impact_ms_per_block` is derived as total impact wall time divided by
`attention_impact_block_count`; it is null when the block count is zero and is
never included in request accounting.

Scopes are stored per raw record in `timing_scopes`:

| Timer | Clock | Scope | Relationship |
|---|---|---|---|
| `request_wall_ms` | CPU wall | complete measured request | inclusive root |
| `tokenization_ms` | CPU wall | tokenizer call | exclusive child |
| `semantic_encode_ms` | CPU wall | encoder call; may include device waiting internal to encoder | exclusive child |
| `cluster_assign_update_ms` | CPU wall | assignment plus immediate Eq.9 update | exclusive child |
| `subsequence_extract_ms` | CPU wall | exact window construction | exclusive child |
| `cache_lookup_ms` | CPU wall | all global-cache lookups and lookup events | exclusive child |
| `hit_selection_ms` | CPU wall | non-overlap selection | exclusive child |
| `attention_impact_ms` | CPU wall | all candidate-block attention reductions, host materialization, per-key means, and history insertion | exclusive child |
| `attention_impact_ms_per_block` | derived | attention-impact wall time divided by candidate-block count | child diagnostic; do not sum |
| `chu_update_ms` | CPU wall | all selected-hit CHU scalar updates and events | exclusive child |
| `cache_policy_ms` | CPU wall | admission scoring, insertion, and eviction for misses | inclusive child |
| `cache_materialization_ms` | CPU wall | tensor-to-cache-entry copies during admitted insertions | exclusive child of cache policy |
| `pbr_update_ms` | CPU wall | configured interval check and any triggered PBR update | exclusive child |
| `prefill_wall_ms` | CPU wall | model forward plus final CUDA-event resolution | exclusive top-level request region |
| `prefill_gpu_ms` | CUDA event | one eager OPT forward including all transformer work and mixed QKV | inclusive child of prefill wall |
| `prefill_host_overhead_ms` | derived | prefill wall minus prefill GPU | child diagnostic; do not sum |
| `mixed_qkv_execution_ms` | CUDA events | sum of disjoint patched Q/K/V calls: native PEFT projection for fresh rows plus cached retrieval/merge | exclusive child of prefill |
| `correctness_reference_ms` | CPU wall plus synchronized GPU forward | separate native reference request | outside `request_wall_ms` |
| `quality_diagnostics_ms` | CPU wall | logit host materialization and error/KL/argmax reductions | outside `request_wall_ms` |

`qkv_execution_ms` is a schema-compatible alias of
`mixed_qkv_execution_ms`; never sum the two.

`prefill_host_overhead_ms = prefill_wall_ms - prefill_gpu_ms` is a diagnostic
approximation of Python, hook, kernel-launch, and final synchronization overhead
associated with the forward. It is not a measurement of pure CPU compute.

`request_wall_ms` closes after the selected execution mode and its SemCache
control/cache-policy path complete, before logits are materialized for benchmark
comparison. The correctness-reference forward is executed by the runner before
the measured request. Timed engine calls reject the legacy inline-reference
option unless an externally computed `baseline_logits` tensor is supplied.
Quality diagnostics remain in every measured record but neither the reference
forward nor logit comparison is part of request latency.

`base_qkv_projection_ms`, `lora_qkv_projection_ms`, `attention_ms`, and
`remaining_transformer_ms` are null: PEFT's native projection combines base and
LoRA work, while splitting attention/remaining work requires more intrusive
module rewriting. Inclusive parents must never be summed with their children;
in particular, do not add cache materialization to cache policy or mixed QKV to
prefill.

## Request timing coverage

Raw rows include `timing_accounted_ms`, `unaccounted_request_ms`, and
`timing_coverage_ratio`. The accounted value sums only these mutually exclusive
top-level request regions:

`tokenization_ms + semantic_encode_ms + cluster_assign_update_ms +`
`subsequence_extract_ms + cache_lookup_ms + hit_selection_ms + prefill_wall_ms +`
`attention_impact_ms + chu_update_ms + cache_policy_ms + pbr_update_ms`.

Then:

`unaccounted_request_ms = request_wall_ms - timing_accounted_ms`

`timing_coverage_ratio = timing_accounted_ms / request_wall_ms`.

The residual is intentionally signed and is not forced to zero. The calculation
does not add `cache_materialization_ms` because it is inside `cache_policy_ms`,
does not add `prefill_gpu_ms` because it is inside `prefill_wall_ms`, does not add
mixed/QKV timing because it is inside `prefill_gpu_ms`, and does not
add `correctness_reference_ms` or `quality_diagnostics_ms` because both are
outside the request.

## Memory and communication

Before every measured request, CUDA peak statistics are reset. After it,
`torch.cuda.max_memory_allocated()` and `max_memory_reserved()` are recorded.
Immediately before reset, after synchronizing prior work, the benchmark records
`cuda_allocated_before_bytes` and `cuda_reserved_before_bytes`. It also emits
`incremental_peak_allocated_bytes` and `incremental_peak_reserved_bytes` as the
corresponding peak minus baseline (clamped at zero).
`model_weight_memory_bytes` is the sum of model parameter storage. These are
local-process CUDA allocator metrics, **not** the paper's total ES/UD system
memory and not paper-equivalent.

Environment and raw records also identify `physical_cache_storage_device` from
the runtime engine and derive `cache_transfer_path` from that device and the
actual model execution device. The current CPU-cache/CUDA-execution setup emits
`cpu` and `cpu_to_cuda_on_reuse`; this transfer/merge cost is intentionally not
optimized away in M8.

Communication savings remain analytical. Saved elements and bytes use the M7
paper-style accounting. If `--bandwidth-gbps` is supplied, transfer time is
only `bytes / bandwidth`; the record is labelled `ANALYTICAL / SIMULATED` and
is never added to measured GPU or wall time.

## Outputs

- `results/m8/inference_raw.jsonl`: one complete schema-stable record per
  measured repetition, mode, and query; unavailable timers are JSON null.
- `results/m8/inference_summary.csv`: count, mean, p50, p95, population std,
  min, and max for every timing field, grouped by experiment/model/mode/query.
- `results/m8/inference_environment.json`: host, CUDA/PyTorch, exact resolved
  revisions, encoder, adapter, dtype, and repetition configuration.
- `results/m8/lora_training.json`: separate adapter-training result.

Quality fields remain attached: maximum absolute and relative L2 logit error,
last-position KL (baseline to measured), and last-token argmax agreement.
Argmax agreement alone is never treated as a safety claim.

Warmup means GPU/kernel/runtime warmup, not persistent cache priming. Each
warmup executes a complete fresh trace using a newly reconstructed SemCache
state, after which that engine is discarded. Every measured trace then receives
another newly reconstructed state. Consequently warmup hits cannot contaminate
any measured trace. Native warmups are stateless apart from normal device/runtime
kernel initialization.

All modes share the same in-memory model and controlled LoRA adapters. Each
request explicitly selects the same adapter, uses the same tokenizer and text,
and verifies an identical token-ID SHA-256. Model/tokenizer revisions, dtype,
seed, eager-attention selection, adapter name, and token hash are emitted in raw
records. Only control-plane execution and physical QKV reuse differ by mode.

## LoRA training benchmark

The training script creates one rank-8 adapter targeting exactly `q_proj`,
`k_proj`, and `v_proj`, then runs one short deterministic synthetic epoch. It
records synchronized step times, epoch/total wall time, sample/token throughput,
CUDA peaks, trainable parameters, and serialized adapter bytes. Dataset size,
sequence length, batch size, optimizer rate, gradient checkpointing, and mixed
precision are explicit. Checkpointing and fp16 autocast are opt-in and recorded
as **REPRODUCTION_CHOICE**. This is systems timing, not convergence or accuracy.
It does not train or extrapolate 50 adapters, and 6.7B training is unsupported.
`total_training_wall_s` remains inclusive of every optimization step, including
first-step startup overhead. `first_step_ms`, `steady_state_step_mean_ms`, and
`steady_state_step_p50_ms` provide a diagnostic split only; steady state drops
exactly the first step and does not alter total time or throughput.

## Limitations

The RTX 3090 differs from the paper's A100 80GB. The workload is a controlled
microbenchmark, not MultiWOZ/CoQA/SNIPS, and no absolute-paper-latency claim is
made. CPU scopes include Python and event-emission overhead. TinyBERT placement
affects semantic wall time. CUDA allocator peaks exclude other processes and
system/device memory outside this process. Cross-user reuse remains approximate
and `safe_reuse_claimed=false`.
