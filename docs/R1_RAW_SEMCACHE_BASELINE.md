# R1-A: Raw SemCache prefill baseline

R1-A is an independent controlled OPT/PEFT microbenchmark. It does not change
M1–M9 or C0–C9 code, policy, or historical outputs. The implementation was
prepared on local branch `exp/raw-semcache-r1`, based on `555de0c` (c9-B).
The local checkout is `/home/songsong/SemCache`; `/data/khuss/repos/GP_semcache`
is the separate SERAPH execution checkout.

**No GPU latency has been measured during local implementation. No safety or
validated acceleration claim follows from a cache HIT.** This is physical
TOTAL Q/K/V projection reuse during full prefill, not KV-prefix reuse, whole
prefill skipping, or generation. Attention, output projection, MLP, and the
remaining transformer still execute.

## Code audit and reuse decisions

|Path|Execution and relevant costs|
|---|---|
|M8 `metrics.inference.native_request`|Native model, `use_cache=False`, `output_attentions=True`; adapter setup is outside its old request timer. R1 uses its own native boundary to include adapter activation consistently with the engine.|
|`SemCacheEngine.query`, lookup-only|TinyBERT, cluster observation, w3 extraction, exact lookup, nonoverlap selection, mixed executor with no reused rows, actual attention impact, CHU/admission/eviction/PBR. It is not a lookup-only microkernel.|
|`SemCacheEngine.query`, physical|Same frontend/policies, with raw cached TOTAL Q/K/V supplied to the mixed executor. Cold misses still use the mixed executor with full fresh rows.|
|`edgelora.mixed_projection.mixed_projection_path`|For each layer and role, gather fresh positions, invoke native PEFT once on only those positions, allocate/scatter output, loop over cached spans and copy tensors. Passive captures stay on execution device until admission.|
|`CacheEntry.from_tensors`|Compact detached copies; CPU is the original default. CPU admission entails D2H plus ownership copies. GPU storage is an explicitly labeled diagnostic.|
|`models.qkv_injection.qkv_injection`|Correctness hook after the original GEMM. Not used to establish compute savings.|
|`cache.attention_impact.PaperRowL2SumReducer`|Per window and layer, double precision reduction with finite checks and host `.item()`. Potentially many synchronization points; default semantics preserved.|
|C9-A runtime|Different prefill boundary and `use_cache=True`. Source build constructs compressed alternatives and warms codecs even when one target condition is RAW. It cannot be imported as a raw-only R1 source builder.|

R1 calls the existing engine and policies unchanged. `r1_runtime.engine_request`
temporarily binds the engine's mixed-executor and request-close functions and
restores them even on exceptions. This runner is single-threaded and must not
share its model/engine concurrently with another application.

The minimal pass calls the existing mixed executor. The detailed pass has an
isolated implementation of the same gather/native/scatter algorithm with
subspan timers. Optional tiny-model tests compare its outputs against the
legacy executor and observe actual base/LoRA A/LoRA B input row counts, including
zero-hit, partial-hit, and all-hit cases. Until those model tests run on SERAPH,
execution equivalence remains unverified locally.

## Modes and invariant settings

* `FULL_RECOMPUTE`: native OPT/PEFT, no cache/frontend or policy.
* `LOOKUP_NO_REUSE`: complete SemCache frontend/policy, zero physical reuse.
* `RAW_PHYSICAL_REUSE`: original exact-w3-within-cluster selection.
* `POSITION_ALIGNED_DIAGNOSTIC`: separate mechanistic control requiring the
  entire prompt token sequence, source adapter/user, and source position to
  match. Never pooled with original SemCache results.

`RAW_ORACLE_DIAGNOSTIC` is optional in the research plan and is not implemented
in R1-A. There is no forced-hit or policy-bypass path.

One loaded model, tokenizer, eager attention implementation, dtype, and pair
of deterministic **synthetic, untrained** PEFT adapters are shared. All modes
use the same per-request adapter, token IDs, `use_cache=False`,
`output_attentions=True`, eval/inference mode, and no `generate()` calls.
Explicit model/tokenizer/encoder revision options are available; resolved
revisions, fixture seeds, local code hashes, GPU/driver and software metadata
are saved. This is not a trained C9 adapter performance reproduction.

Serving with `use_cache=True`, disabling attention outputs, or using another
attention implementation requires a separate future experiment with all modes
changed together. R1-A deliberately exposes no option that silently alters
these invariants. The current comparison does not quantify the isolated cost
of `output_attentions=True` against an attentions-disabled serving baseline.

## Pairing, preconditions, and correctness

The M8 natural-text construction supplies requested lengths 32/64/128/256;
actual tokenizer lengths are recorded and may differ slightly. Each pairing
group has four conditions: cold miss, same-user exact repeat, cross-user exact
repeat, and unrelated text. For every non-cold mode/request, a fresh engine
receives exactly one common source query using lookup-no-reuse, then the target.
Each condition is independent, rather than inheriting previous target updates.
The cold condition measures first admission in its own labeled scope.

Source forward/impact/policy/materialization and first admission are saved in
`source_build`, outside target timing. Source construction always uses the same
detailed path, irrespective of target pass. Model/adapter initialization and
encoder initialization are separate manifest values. Precomputed anchor
encoding is setup, not a per-target semantic-encoding replacement.

FULL has no source cache build. A common untimed native reference is executed
for each pairing group before modes; repeated warmups and mode-order rotation
reduce ordering bias but do not eliminate GPU thermal or allocator effects.
These remain controlled reconstructed-state microbenchmarks, not a production
arrival trace. Default warmups are 3 and measured repeats 10 per configuration;
only `--smoke` permits smaller counts. Mode order rotates; pass/storage order
alternates per repetition. Warmups are saved but excluded from summary statistics.

The original key remains `(cluster_id, exact_token_ids)`, with window size 3.
A window is three tokens; multiple nonoverlapping windows can cover most of
the prompt. Logical HITs and executed spans/tokens are separate fields.
Repeated subsequences may fetch the same source block at different target
positions. Exact token equality alone does not ensure equal contextual QKV.
Source/destination positions, adapter identities, duplicates, and cache
residency signatures are recorded. Across CPU/GPU and minimal/detailed passes,
the runner requires matching source residency, selected-span provenance, and
input token hashes before accepting combined reports.

Projection records must cover every layer/role, with exactly the complement
of reused positions passed to native projection. All-layer/role skipped rows
are `reused_tokens * num_layers * 3`; this is not the same unit as token coverage.
FULL minimal has no added projection hooks, so its observed row records are
null, not invented measurements.

Logits are materialized and compared outside the closed target request.
Outputs include max/relative logit errors, baseline-to-result KL, final argmax
agreement and `torch.allclose` at explicit tolerances (defaults atol/rtol 1e-3).
Natural reuse may fail parity and is still reported honestly. Cold, lookup,
FULL or strict-diagnostic parity failures produce `CORRECTNESS_FAILED` and a
nonzero exit. No nonzero validated strict reuse yields `NO_VALIDATED_REUSE`;
zero-hit self-parity is not evidence for reuse correctness. There is no text
quality or downstream-task claim.

## Timing scopes and instrumentation limits

All request starts and ends synchronize the selected CUDA device. Adapter
activation, tokenization/input creation, frontend, prefill and applicable
cache work are inside request latency. Reference forward, comparisons, report
serialization, memory inspection, and source construction are outside it.

|Metric|Scope / interpretation|
|---|---|
|`request_wall_ms`|Synchronized complete target request through policy, before diagnostic logit CPU transfer.|
|`prefill_wall_ms`, `prefill_gpu_ms`|Nested in request. Detailed pass resolves prefill end before policy. Minimal engine prefill fields are null because its legacy host boundary is asynchronous. Native minimal prefill wall ends at its request completion.|
|`semantic_encode_ms`, `cluster_assign_update_ms`|Existing engine host intervals; encoder includes its own host vector materialization.|
|`subsequence_extract_ms`, `cache_lookup_ms`, `hit_selection_ms`|Existing Python extraction/matching/selection intervals.|
|`attention_impact_ms`, `chu_update_ms`, `pbr_update_ms`|Existing policy semantics and host synchronization costs.|
|`cache_policy_ms`|Admission, eviction and materialization; `cache_materialization_ms` is nested within it, not additive.|
|`mixed_executor_host_ms`, `mixed_executor_gpu_ms`|Detailed gather/native/scatter path; nested in prefill. GPU events include stream idle gaps from host submission, not just kernel time.|
|`native_qkv_host_ms`, `native_qkv_gpu_ms`|Native PEFT projection on fresh rows, including base plus adapter computation. FULL uses passive hooks. Not a pure cuBLAS kernel counter.|
|`cache_fetch_host_ms`, `cache_fetch_gpu_ms`|Per cached tensor `.to(output)`, including dtype/device handling. CPU→GPU bytes/call counts are separately recorded. GPU-resident same-device fetch can be a no-op.|
|`index_transfer_*_ms`, `tensor_assembly_*_ms`|Fresh index transfer, then gather/allocation/scatter. Nested in mixed execution.|

**Never add nested GPU events to prefill/request, CPU host times to GPU times,
or the legacy `qkv_execution_ms` alias to `mixed_qkv_execution_ms`.** Stream
intervals include Python-induced idle time. Transfer host intervals can include
blocking pageable-memory copy behavior. No per-subspan explicit synchronization
is introduced; events resolve after the request boundary synchronization.

`minimal` means no additional inner CUDA events/native hooks; existing engine
CPU timers, event logs, audits and algorithm-required synchronization remain.
It is not instrumentation-free serving. Detailed execution is a separate
instrumented implementation, so `instrumentation_overhead.json` measures both
added timers and their Python control overhead. The paired difference is a
diagnostic observation, not a correction automatically subtracted from latency.

## Raw-only isolation and memory

R1 never imports the C9 runtime, constructs codec/storage backends, or invokes
quantization, compression, or transport. A fail-closed import guard blocks the
repository CacheGen runtime and known codec/quantization package roots. It
rejects already-loaded forbidden modules and records blocked attempts. Model
quantization is rejected, PEFT projections retain unquantized validation, and
resident entries are checked for floating raw tensor ownership/device. The
reported zero codec calls is evidence from this restricted execution path,
not a system-wide Python profiler counting unrelated libraries.

`--cache-devices cpu,cuda` runs `RAW_CPU` and `RAW_GPU_DIAGNOSTIC` separately
with equal byte budgets. The name is a storage condition applied to the whole
comparison, including its matched FULL. GPU storage does not replace CPU
defaults. Signatures include source positions, adapter/prompt provenance,
tensor shapes/dtypes and resident byte accounting. No cross-device result is
accepted if selected hits or source residency change.

After model loading, a conservative preflight screens attention/reducer
workspace, activations, logits, twice the GPU cache budget (temporary admission
overflow), and a 512 MiB reserve against free plus reusable allocator bytes.
This is not an OOM guarantee. Model-load OOM, failed preflight, and runtime OOM
write a FAILED manifest with reason and active pairing group; already committed
checkpoints survive. Peak allocated/reserved GPU memory and resident cache
bytes are recorded per target. CPU resident cache capacity does not describe
peak process RSS. 512+ tokens are rejected in this stage, even on 24 GB GPUs.

## Local tests and SERAPH commands

Local validation, no model dependencies required:

```bash
python3 -m py_compile scripts/69_run_r1_raw_semcache.py src/semcache/metrics/r1.py src/semcache/metrics/r1_runtime.py tests/test_r1_raw_semcache.py
python3 -m unittest discover -s tests -p test_r1_raw_semcache.py -v
python3 scripts/69_run_r1_raw_semcache.py --help
```

On the implementation host: 9 utility/schema/CLI tests passed, 2 tiny-model
tests skipped because torch/transformers/peft are unavailable. pytest is also
unavailable. Syntax compilation and CLI help passed. OPT-125M, OPT-2.7B, CUDA,
legacy before/after timings, and CPU/GPU tensor execution remain **unexecuted**.

On SERAPH, use its existing compatible torch/transformers/peft environment and
pre-cached OPT and TinyBERT checkpoints. Loading is local-only by default.
Run from `/data/khuss/repos/GP_semcache`. Start with the tiny CPU tests above,
then OPT-125M smoke:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 scripts/69_run_r1_raw_semcache.py \
  --model facebook/opt-125m --device cuda:0 --dtype float32 \
  --prompt-lengths 32 --cache-devices cpu --instrumentation both \
  --warmup-runs 1 --measured-runs 2 --smoke \
  --output-dir results/r1/opt125m_cpu_smoke
```

Then run the OPT-2.7B CPU baseline in increasing length order:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 scripts/69_run_r1_raw_semcache.py \
  --model facebook/opt-2.7b --device cuda:0 --dtype float16 \
  --prompt-lengths 32,64,128,256 --cache-devices cpu \
  --warmup-runs 3 --measured-runs 10 --instrumentation both \
  --output-dir results/r1/opt27b_raw_cpu
```

CPU/GPU storage diagnostic with independently matched FULL/LOOKUP/PHYSICAL:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 scripts/69_run_r1_raw_semcache.py \
  --model facebook/opt-2.7b --device cuda:0 --dtype float16 \
  --prompt-lengths 32,64,128,256 --cache-devices cpu,cuda \
  --cache-capacity-mib 256 --warmup-runs 3 --measured-runs 10 \
  --instrumentation both --output-dir results/r1/opt27b_storage_diagnostic
```

For tighter provenance add resolved `--revision`, `--tokenizer-revision`, and
`--encoder-revision` to the initial command and preserve them on restart.
Resume with the exact same command plus `--resume`. Changed arguments, code
hashes, resolved runtime/model identity are rejected. Warmups rerun after
restart; completed measured groups are skipped. Incomplete groups rerun from
fresh source state. Output directories must be new unless explicitly resuming.

The old M8 runner may additionally be run unchanged, with an isolated suffix,
to provide a legacy reference; its request/state boundaries differ from R1:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 scripts/30_run_m8_inference_timing.py \
  --model facebook/opt-2.7b --device cuda:0 --dtype float16 \
  --warmup-runs 3 --measured-runs 10 --prompt-lengths 32,64,128,256 \
  --position-aligned-diagnostic --experiment-suffix r1_legacy_audit
```

## Outputs and remaining questions

Each R1 output directory contains `manifest.json`, atomic `checkpoints/*.json`,
`raw.jsonl`, `summary.csv`, `summary.md`, `correctness.json`, and
`instrumentation_overhead.json`. Checkpoints preserve paired rows and source
build costs. Summary groups never pool cold/steady targets, diagnostics,
lengths, storage, or instrumentation. They report mean/p50/p95 request and
component times, coverage, parity counts, and matched **condition minus FULL**
latency deltas. Negative deltas alone do not validate reuse quality. Failed
natural parity remains visible; no automatic speedup claim is emitted.

R1-B should first inspect: CPU fetch versus GPU cache, native projected rows
and QKV time versus assembly/validation loops, TinyBERT/lookup overhead,
attention-impact synchronization, admission copies, and instrumentation
perturbation. C9's observed performance order cannot be explained numerically
without executing these measurements and matching model/adapter/workload
boundaries. An attentions-disabled native-serving reference and a carefully
matched `use_cache=True` study can follow separately. QMSum/Qasper can later
provide case dictionaries at the workload boundary while preserving token
hashes and pairing; no dataset conversion or Qwen support is implemented here.
