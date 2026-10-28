# R2-C1 Long-context RAW QKV bottleneck and break-even diagnosis

Initial checkout: `exp/raw-semcache-r2`, HEAD
`85cf8f4da5d36c5c7ed35dd03f2135710891cf1a`, clean working tree. Frozen
69/72/80/82/83/89 runners, R1-B3 runtime, Engine, mixed/contiguous executors,
source/selection modules and relevant tests/docs were inspected. No frozen code,
test, result or Slurm file is changed. Status is IMPLEMENTED_LOCAL_ONLY;
GPU validation NOT_RUN. Historical findings are not new C1 measurements.

## Three valid conditional request paths

One OPT/PEFT instance and one existing TinyBERT encoder serve all modes. The C1
Runtime subclasses R2-A CostRuntime and delegates source construction and
requests to frozen R1-B3/R2-A. No new projection, matching or policy algorithm.

| C1 mode | Frozen implementation | Work included |
|---|---|---|
| FULL_NATIVE | native_request via CostRuntime | tokenizer, adapter/eval, native Full Forward and LM Head |
| SEMCACHE_NO_REUSE | request(mode=lookup) | real encoder/cluster, exact-w3 lookup, selection, native-only mixed wrapper, Forward, impact, executed cache policy/CHU/PBR |
| RAW_SEMCACHE_QKV | request(mode=contiguous) | same Engine control, fast position-aware occurrence selection, mapping/snapshot validation, contiguous or fallback QKV assembly, fresh projections, remaining full Transformer and LM Head |

All use user_a, controlled rank-8 QKV LoRA, the same model/tokenizer/IDs,
dtype/device/eager backend, **use_cache=False, output_attentions=True**, and
native all-token logits [1,L,V]. No last-token head or Prefix KV cache is added.
Actual model/LM Head root calls and shapes are checked, not inferred from names.

The RAW occurrence catalog bypasses production Admission eligibility. It remains
**ORACLE_DIAGNOSTIC**, safe_reuse_claimed=false, even though actual semantic
embedding, clustering, lookup and Engine policies execute. This is controlled
same-adapter/exact-prefix/position reuse, not natural multiuser SemCache service
or validated cross-user reuse. No-Reuse is not a pure frontend-only control:
it also executes the existing native-only mixed wrapper and Engine policy.

## Boundary audit

Primary clock is the existing synchronized R1 CPU request_wall_ms, ending after
Forward and (for Engine modes) impact/policy work. Full and Engine paths both
exclude Logit CPU transfer and quality. No profiler or added hooks in minimal
timing. Tokenization and the operations in the table are inside each request.

The frozen harness constructs a fresh Engine deepcopy and resets Catalog policy
state **outside** Engine request timing. Controller allocation, post-request
mapping/row audits and guard/hash checks are also outside. C1 preserves this
boundary. wrapper_invoke_wall_ms separately exposes the broader invocation
including clone/control/postchecks/CPU transfer, but is never substituted for
the primary clock. These are conditional R1 request comparisons; do not call
them a complete serving or whole-harness E2E measurement.

Source-only detailed SemCache prefill/capture/hash instrumentation, catalog
copy/build, fast index, backing descriptor and initial MAX mapping preparation
are separate. The source prefill label is an instrumented SemCache source cost,
not minimal native prefill. No build cost is hidden in the target request or
assumed absent from a system deployment. No amortization model is implemented.

Paired observations: delta_full_raw=Full−RAW, delta_miss_raw=NoReuse−RAW,
delta_full_miss=Full−NoReuse; positive means the second path is faster. These
are request contrasts, not exclusive causal component costs.

## Prefix KV reference and output parity

R2-B3.5 is historical reference only: use_cache=True, attention OFF, one-token
head and whole-prefix decoder skip, with Cache clones. C1 is use_cache=False,
attention ON, all-token head, full remaining Transformer/MLP/attention execution,
QKV-only row skipping, source Catalog and Engine policy. Token fixtures may come
from the same text constructor but historical actual IDs/weights/timers are not
asserted identical. No fresh Prefix bridge is executed or historical samples
pooled. A native Prefix KV ratio is never labeled a RAW QKV speedup.

Within C1 all three output/head policies are matched at L rows. This does not
make C1 output-matched to B3.5, nor remove the disclosed harness boundary limits.

## Workload, semantic scope and coverage

Smoke: OPT-125M FP32, 32/64. Pilot starts 512/1024; optional 1536/2048 requires
matching completed C1 base pilots first. Actual model max_position_embeddings
is checked, including BOS/special IDs in the exact tokenizer result. Requests
are never truncated to another OPT length. Existing controlled_length_trace and
its source/target IDs are reused; exact roundtrip and requested length must agree.

The inherited TinyBERT encoder has a separate tokenizer and max_length=512 with
truncation enabled. C1 records original semantic token count, cap and truncation
flag explicitly; OPT still processes all requested tokens. Semantic processing
cost need not scale beyond that encoder cap. No silent change to encoder or
clustering policy is made for long context.

Default requested coverages are 0,.5,max; optional .75/.875 can be supplied.
The same deterministic exact-w3 occurrence selector chooses earliest valid
nonoverlapping windows. Each window is three tokens; feasible MAX is at most
floor(L/3)*3, not automatically 100%. At L1536 that bound permits all tokens and
zero native fresh QKV calls; the audit handles this separately from no-reuse.
The theoretical bound is not a measured attainment claim.

Record actual positions/windows, source/target token and absolute position,
prefix/adapter evidence, actual native/skipped rows, mapping cache hit, setup and
selection host times, contiguous versus fallback groups/copy counts and temporary
memory. Requested target coverage is never relabeled as attained coverage.

Inherited executor byte counters include logical slice/view row volumes; they
are not measured temporary allocator peaks. Actual Torch allocated/reserved peaks
and profiler allocator events are labeled separately. No frozen counter is
silently redefined.

## Structural and numerical gates

Source capture hashes are compared to actual owned Catalog backings. Index and
descriptor guards protect provenance, reference/offset/stride/dtype/device and
tensor versions. Full parameter hashes and canonical backing contents are checked
outside timers before/after case work. Model/adapter/backend/eval state is guarded.

Untimed root/head hooks prove one native model and one all-token LM Head call
per mode. LoRA B child hooks (not forbidden QKV-parent hooks) prove actual fresh
row counts and active scaling/adapter. Reused TOTAL QKV includes the captured
source adapter contribution. If all rows are cached, no target LoRA B/native
projection call is expected. Actual skipped/native audit must match unique
selected positions, exact token IDs, source positions and prefix/adapter checks.

An untimed scoped observer reads existing passive executor outputs: it hashes
one QKV projection at a time and compares reused rows against canonical source
gathers. It never performs another projection or saves tensors to disk. The same
selection through the frozen Legacy window executor must match Contiguous QKV
hashes, logits, rows and selection hash exactly. Gate overhead is discarded from
primary timing; these observers are absent during minimal requests.

Full-vs-RAW and Full-vs-NoReuse logits use B3's explicit elementwise rule
abs(actual−reference) ≤ 1e-3 + 1e-3*abs(reference), with magnitude/relative L2,
violation count, KL and argmax as distinct metrics. FP16 Strict FAIL remains
visible and does not stop a structurally valid diagnostic. Physical/alignment,
canonical mutation or Legacy/Contiguous parity failure invalidates the case.
The strict gate and measured samples both constrain aggregate qualification.

## Capacity and staging

No second OPT model is loaded. Screening records actual model/LoRA and encoder
resident bytes, source TOTAL QKV estimate, engine capacity, source-copy transient,
passive target QKV, assembly workspace, attention/activation/logit reserves and
profile reserve. Python metadata is measured after build where available; it is
not fabricated as a fixed zero. Additional headroom uses actual free/allocated/
reserved memory. Original R1 preflight also runs unchanged and is conservative.
Its rejection is MEMORY_PREFLIGHT_FAILED, not evidence of actual GPU OOM.

Default capacity is 512 MiB (1024-token OPT-2.7B FP16 full source QKV is about
480 MiB by shape accounting). 2048 needs an explicitly larger capacity (about
960 MiB source alone). Changing capacity requires matching new base pilots.
The conservative attention reserves can reject 1024+ before real execution;
this limitation is not silently bypassed. An estimator adjustment would require
a separately reviewed capacity experiment. No successful long-GPU run is assumed.

Cases are independently executable. OOM/preflight failures retain requested
length, context, traceback and memory-at-failure; later cases continue. Only
failed-case cleanup may clear CUDA cache, outside request timers. Successful
source/index tensors are released between lengths, without timed cache clearing.
No length/backend/coverage is silently substituted.

Latency/profile require exact length+coverage approvals from completed C1 pilots
on matching code/model/weights/encoder/GPU/storage/capacity/seed/settings, followed
by actual source-content/token identity verification. >1024 pilots additionally
require successful 512/1024 approvals for selected coverages. Use small explicit
subsets when capacity is unsupported; earlier failure provenance stays intact.

## Statistics and hypotheses

Three separate process runs, ≥3 warmups, ≥20 measured three-mode pairs per case.
Seeded counterbalanced Latin mode order; reproducible shuffled case order for
latency; ascending pilot lengths. Raw samples are checkpointed per completed pair.

Session means get equal weight. Preserve individual statistics, wins and outliers,
between-session variance and exploratory session-mean bootstrap CI. Never treat
60 pairs as 60 independent processes. Reject different endpoints/hardware/code,
mixed profiler/pilot data, incomplete or duplicate samples and changing coverage.
Input GPU execution status is explicitly aggregated; analysis_process records
gpu_inference_performed=false and never inherits a misleading NOT_RUN default
as the status of completed GPU input sessions.

H1 compares same-session Full costs across measured lengths at fixed requested
coverage. H2 compares RAW wall across attainable reuse at fixed L. Positive
consistent changes in ≥3 verified sessions support only the observed net trend;
kernel/selection/assembly causal shares remain unresolved. H3 reports reproduced
positive conditions only with all session means positive, positive exploratory
CI and verified environment. Otherwise positive-mean-only or
NO_OBSERVED_BREAK_EVEN is a valid result. No interpolated threshold or extrapolated
longer speedup. Correctness-qualified evidence additionally requires strict gates
and samples; it is still an ORACLE conditional diagnostic, not production safety.

## Separate profiling

Target selected ≤2 lengths/coverages per profile invocation. Reuse R2-A's supported
CPU/CUDA activities, native QKV Base/LoRA, attention/FFN/head and Engine ranges.
V2 imports batched_mixed_projection.span at context entry, so R2-A already wraps
its actual gather/conversion/scatter stages. C1 additionally observes actual
GlobalCache.insert policy work. No executor arithmetic is modified.

CPU self time, attributed device self time, raw kernel duration/count, memcpy and
CUDA runtime synchronization activity are separate. Parent/child ranges, CPU/GPU
clocks and overlapping activity sums are never added into a fictitious request
total. Unknown activity remains unknown. Profiler output must equal a plain
reference; missing profiler is explicit. Memory/copy/mapping counts are retained.
No cuBLAS algorithm, occupancy or CPU-dominance causal conclusion is inferred.
Attention module ranges include their QKV children. Reduced inclusive duration
does not prove attention computation was skipped: C1 still processes L-token
attention/MLP, unlike Prefix KV suffix-only decoder execution.

## CLI (SERAPH offline)

```bash
conda activate semcache
cd /data/khuss/repos/GP_semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/90_run_r2c1_long_context_qkv.py --phase smoke \
  --model facebook/opt-125m --dtype float32 --prompt-lengths 32,64 \
  --coverages 0,.5,max --device cuda:0 --cache-device cuda \
  --output-dir results/r2/c1/smoke01

python scripts/90_run_r2c1_long_context_qkv.py --phase pilot \
  --model facebook/opt-2.7b --dtype float16 --prompt-lengths 512,1024 \
  --coverages 0,.5,max --capacity-mib 512 --device cuda:0 --cache-device cuda \
  --warmup-runs 1 --measured-runs 3 --independent-run-id pilot01 \
  --output-dir results/r2/c1/pilot01

# Only after every selected case has a validated pilot.
for run in 1 2 3; do
  python scripts/90_run_r2c1_long_context_qkv.py --phase latency \
    --model facebook/opt-2.7b --dtype float16 --prompt-lengths 512,1024 \
    --coverages 0,.5,max --capacity-mib 512 --device cuda:0 --cache-device cuda \
    --pilot-inputs results/r2/c1/pilot01 --warmup-runs 3 --measured-runs 20 \
    --seed 42 --order-seed "$((4100 + run))" --independent-run-id "run${run}" \
    --output-dir "results/r2/c1/latency_run${run}"
done

python scripts/90_run_r2c1_long_context_qkv.py --phase analyze \
  --inputs results/r2/c1/latency_run1 results/r2/c1/latency_run2 results/r2/c1/latency_run3 \
  --output-dir results/r2/c1/aggregate01

python scripts/90_run_r2c1_long_context_qkv.py --phase profile \
  --model facebook/opt-2.7b --dtype float16 --prompt-lengths 512,1024 \
  --coverages max --capacity-mib 512 --device cuda:0 --cache-device cuda \
  --pilot-inputs results/r2/c1/pilot01 --max-trace-mib 64 \
  --output-dir results/r2/c1/profile01
```

If only 512 is validated, explicitly use --prompt-lengths 512 consistently in
latency/profile. For optional longer capacity, first repeat base pilots with
--capacity-mib 1024 in a new directory; then --phase pilot --prompt-lengths
1536,2048 --capacity-mib 1024 --pilot-inputs <matching-base-pilot>. Extend only
when those cases are validated. No automatic full sweep or Slurm file is created.
Downloads are disabled unless --allow-download is explicitly supplied. Pin model,
tokenizer and encoder revisions for exact reproduction. Do not reuse output dirs.

## Artifacts and local verification

Manifest/environment, capacity_pilot (pilot), workload_audit, coverage_audit,
correctness_and_gates, timing_boundary_audit, latency_raw_pairs.csv/.json,
latency_session_summary.csv; analyze adds latency_aggregate_summary.json,
break_even_decision.json and decision_summary.json. Profile adds
bottleneck_breakdown.csv, profiler_kernel_summary.csv, profiling_status.json and
profiler_trace/*.json. Unsupported measurements remain null/status-marked.
Global failure retains full traceback and already checkpointed artifacts.

```bash
PYTHONPATH=src:tests python3 -m unittest test_r2c1_workloads test_r2c1_analysis \
  test_r2c1_benchmark test_r2a_runtime test_r2a_profiler \
  test_r1b2 test_r1b3_fast_selection test_r1b3_batched_executor \
  test_r1b3_b21_mapping -q
python3 scripts/90_run_r2c1_long_context_qkv.py --help
git diff --check
```

Tests use synthetic control/timing data, never GPU findings. Native OPT/LoRA audit
and frozen tensor executor tests are dependency-gated. The real semantic/model
three-mode GPU Smoke remains NOT_RUN without local Torch/Transformers/PEFT.

Local verification on 2026-10-28: 175 related tests, 130 passed, 45 skipped,
zero failures (new C1: 15 tests, 14 passed and one dependency-gated native audit
skipped). Syntax compilation and CLI validation passed. Synthetic fixtures and
historical GPU reports are not newly collected GPU evidence.
