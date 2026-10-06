# C9-A measured local prompt latency

C9-A measures **cache lookup through next-token logits**, using one prompt
forward per repetition (`use_cache=True`, as in C8-B greedy prefill). It does
not run autoregressive generation, teacher-forced continuation, BLEU, network
analysis, or transport compression. Frozen32 remains a selection-based
controlled trace, not an untouched final test cohort.

The seven conditions are FULL_RECOMPUTE and B2/B8 crossed with RAW_QKV,
KV_COMP, and Q24_KV_COMP. B2 is 2,949,120 bytes; B8 is 11,796,480 bytes.
FULL is measured once per target and reused as the reference for both budgets.
The exact-w3-within-semantic-cluster HIT and cached TOTAL Q/K/V payload stay
fixed. The existing mixed path skips three native rows per role per layer on
HIT; MISS executes the normal native model path.

Before loading the model, the script verifies the frozen SELECT_Q24 artifact,
its pinned Q24/K20/V16 profiles and upstream evidence, the canonical B3 model,
adapter, workload and selection chain, every required C8-A output binding, and
the completed C8-B manifest and HIT audit. C8-B must support Q24 for C9 and
reproduce A's resident states and HIT vectors. C9 reconstructs the same byte-LRU
admissions from A's variable per-entry accounting, then requires identical
actual physical bytes, eviction decisions, retained source identities, LRU
snapshots, and every target HIT/MISS decision. Expected counts are validated
against the artifacts: B2 has 2/4/13 HITs and B8 has 8/18/32 HITs. Saved vectors,
not these aggregate counts, govern each request.

All 32 source TOTAL-QKV blocks are captured once. Compact selected rows are
copied to CPU; all three physical representations are constructed from the
same payload. Encoded objects are shared between budgets. RAW owns compact
FP16 Q/K/V; KV_COMP owns raw Q and compressed K/V; Q24 owns compressed Q/K/V
without raw resident tensors. Decoded tensors exist only in temporary HIT
views. Frozen profiles are read-only and guarded against runtime fitting.
Shared profiles and temporary working copies are not charged per entry.
Source forward, CPU capture, storage preparation/encode, and admission are
timed separately, without disk I/O or model loading. RAW preparation measures
compact tensor ownership copies, without a fictitious codec call. Source costs
are secondary and never included in primary target latency. Scenario source
totals reuse shared costs; summing them would overcount construction work.

There are exactly five untimed target warmup forwards, alternating native and
mixed-native paths. Before timed encoding, the first real captured source block
also warms raw preparation and the frozen encode/decode paths without another
source/model forward or an admission. These codec warmup calls are excluded
from source-build timings and recorded separately. There is no shape/repetition
sweep. Each target is then measured three times per condition, in the same
canonical order: **672 measured prompt forwards, five warmup forwards, and
32 source forwards**. LRU order is restored outside timing before each repeat;
the third lookup advances the canonical trace once. Targets are never admitted.

Wall measurements use `perf_counter_ns` with `torch.cuda.synchronize(device)`
before request start and at next-token-logit availability. Compressed decode
and model-forward boundaries also synchronize; their synchronization cost is
included in the instrumented total. Lookup is a CPU-only logical residency
decision; MISS and RAW have zero decode. HIT decode includes Q+K/V or K/V as
appropriate. Model timing includes adapter activation, prompt input transfer,
the actual model execution, and next-token-logit selection. Validation and
event resolution occur after the total closes.

Passive outer-module CUDA event hooks time native Q/K/V on FULL/MISS, without
altering inputs or outputs or running a second projection. HIT uses the existing
mixed path's CUDA events. Their sum is **nested within model_forward_ms** and
must never be added to model-forward or request totals. The measured request
includes timing-hook/context setup and event recording overhead; this is a
synchronized, instrumented local measurement, not uninstrumented serving
latency. No diagnostic CPU tensor copies occur within target timing.

Each raw repetition must satisfy total = lookup + decode + model + remaining
control overhead. Each target's three repetitions are reduced to a median for
each component. Independent component medians need not sum to the median
total. Across targets, ALL/HIT/MISS subsets report cases, mean, p50, linearly
interpolated p95, and population standard deviation. Empty subsets have null
metrics. FULL comparisons use the same episode identities and matching subset;
speedup is the ratio of matched FULL mean to condition mean. Paired differences
are condition minus reference on the per-target medians, with faster/slower/tied
counts. The predeclared numeric tie tolerance is ten nominal perf_counter
resolution ticks; it is not a statistical equivalence threshold.

The manifest records input/output hashes, model/adapter/profile provenance,
actual GPU/software, synchronization and nesting policy, repetition counts,
and runtime counters. The expected decode counts follow the saved vectors;
MISS cannot decode and no runtime fitting, transport, or quality forwards are
allowed. Partial failures write INVALID reports retaining their primary reason.
A new output root is required; historical results are never overwritten.

The readiness rule requires valid provenance, exact C8 state/HIT reproduction,
synchronized GPU measurements, zero runtime fitting/transport, and consistent
timing decomposition. C9_A_READY_FOR_NETWORK_ACCOUNTING does not mean Q24 is
fastest and does not freeze a system policy. No network time is computed and
C9-B is not launched.

Run with one GPU allocated from `/data/khuss/repos/GP_semcache`:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u scripts/67_run_cachegen_c9a_local_latency.py \
  --adapter-root results/cachegen/c6b3/b1_full \
  --adapter-freeze-decision results/cachegen/c6b3/b1_full_freeze_decision.json \
  --plan-dir results/cachegen/c6b3/multiwoz_plan_v2 \
  --source results/workloads/multiwoz.jsonl \
  --semantic results/workloads/c6b3_multiwoz_history_semantic.jsonl \
  --freeze-decision results/cachegen/c7/q24_freeze/freeze_decision.json \
  --c8a-root results/cachegen/c8/a_capacity_replay \
  --c8b-root results/cachegen/c8/b_quality_b2_b8 \
  --kv-profile results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin \
  --storage-src /data/khuss/repos/GP_semcache/.c6_storage_src/src \
  --device cuda:0 --seed 42 --warmup 5 --repeats 3 \
  --output-root results/cachegen/c9/a_local_latency
```

Artifacts: source_build_latency.csv, source_build_summary.json,
target_latency_raw.csv, target_latency_per_case.csv, target_latency_summary.json,
hit_miss_latency.json, paired_latency.json, manifest.json, summary.md.
Runtime has not been measured locally. Budgeted work includes 201 measured
K/V decodes and 135 measured Q decodes in addition to the prompt/source forwards;
CPU entropy-codec cost and the allocated GPU determine elapsed time. No estimate
based on unmeasured GPU timings is claimed. All inputs must already exist on
SERAPH; model loading is local-only and runtime caches/outputs stay under
`/data/khuss`. This stage creates no shell/Slurm scripts.
