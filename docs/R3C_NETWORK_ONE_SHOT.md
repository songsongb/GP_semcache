# R3-C: controlled TCP conditions and exact-prompt workload crossover

`scripts/100_run_r3c_network_one_shot.py` is an add-only diagnostic extension of
R3-B. Classification is `R3C_APPLICATION_LAYER_NETWORK_DIAGNOSTIC`.
`PRIMARY_PERFORMANCE_ELIGIBLE`, `correctness_qualified_reuse_speedup`,
`actual_remote_network_measured`, `semantic_generalization_validated`,
`trained_adapter`, `cachegen_enabled`, `compression_enabled`, and
`r2d_break_even_validated` remain false, including when every cache gate passes.
The experiment measures synchronized **all-token prefill request wall**, not
TTFT, generation, task quality, remote UD hardware, or natural semantic HITs.

The user reported that SERAPH R3-B job 151857 passed its physical cache equality
gates while Native PEFT strict parity remained FAIL. Those historical artifacts
are unavailable on the implementation machine and are neither loaded nor
verified here. Their latency values are not runtime expectations. Every R3-C
GPU run prepares a fresh independent cohort; it does not reuse 151857 evidence.

## Architecture and preserved interfaces

```text
one isolated 95 native prepare process (exits before ES starts)

persistent bare OPT GPU ES <--real framed TCP--> persistent CPU proxy
                                               <--real framed TCP-->
                                               original CPU-only UD process

all bindings and connection targets: 127.0.0.1
```

The actual `communication.Channel`, `ProjectionSession`, `user_device.serve`,
`cpu_deltas`, R3-B `Experiment.request`, `physical_path`, and `ExactRawCache`
are reused. R3-C subclasses the request coordinator and routes its Channel to
the proxy. It does not replace tensor arithmetic or instrument production
95–99 files. The new UD worker scopes a Channel **timeout-only** constructor
override to 120 seconds; protocol bytes and original FP32 CPU math are unchanged.
The separate UD has hidden CUDA devices, bounded CPU threads/recorded affinity,
actual original A/B weights, no full Transformer, and no CUDA computation.

After calibration, one ES loads one bare FP16 OPT-2.7B model, checks its weight
fingerprint, and retains it across all conditions. It must match the fresh
prepare's GPU UUID, versions, adapter, exact token IDs, endpoint, and snapshots.
No LoRA matrices reside on the ES. The fresh prepare may use any allocated RTX
3090; only identity **within that new allocation/cohort** must match. There is
no GPU-node pin or requirement to match an older run's UUID.

The genuine SemCache components remain R3-B's `CacheEntry`, `GlobalCache`,
`ExactTokenMatcher`/`Subsequence`, admission metrics/policy and owned compact
total-QKV tensors. A 64 MiB ES GPU cache holds complete final Q/K/V outputs.
Model/adapter/token IDs/positions/context/endpoint/dtype/shape/content/provenance
are validated; only exact full-prompt replay can authorize reuse. A MISS sends
one HIDDEN and receives one bundled FP32 DELTAS per actual layer, assembles
`(FP16_base + FP32_delta).to(FP16)`, and admits all 96 outputs after completion.
A full HIT clones cached total QKV during the real decoder forward and bypasses
both base projections and UD work. The remaining decoder and LM head still run.
`SemCacheEngine.query()`, clustering, `w=3` window reuse, cross-context reuse,
CacheGen, learned personalization and natural HIT-rate estimation are absent.

## Link emulator and units

The same independent full-duplex proxy process serves the CPU calibration
session and the model session. It forwards the **original** R3-A prefix, JSON
header and tensor bytes without re-encoding any application frame, sequence,
session, request ID, SHA or tensor layout. Both original Channels independently
validate the data; relay and endpoint frame SHA/byte ledgers are reconciled.
TCP/IP headers, ACKs and retransmissions are outside application wire counts.

Each real application frame is delayed once, after its header is available and
before delivery starts: configured RTT/2 in each direction. The relay streams
the entire frame through independent directional virtual byte clocks. Pacing
includes prefix, JSON and tensor payload. Each frame starts with zero credit;
scheduler/read/hash overshoot may accumulate at most one 16 KiB chunk of
catch-up credit. Credit resets at the next frame and never grows across idle
periods. At most one chunk can be delivered as a catch-up burst. This bounds
bursts while avoiding repeated charging of OS wakeup overhead as link time.
No saved latency is adjusted and no sleep is added after request timing.

Payloads are capped at 2 MiB, JSON headers at the original 64 KiB, streaming
buffers at 16 KiB plus a bounded header per direction, socket buffers requested
at 64 KiB, and relay ledger at 40,000 frames. Kernel buffer doubling/queueing
is not interpreted as measured pure link latency. Frame deadlines are 20 s;
idle and original model Channel deadlines are 120 s. Ordering is preserved.
Profiles change only at drained source/relay count checkpoints between requests.

Mbps means decimal 1,000,000 bits/s, 8 bits/byte. Time is ms and sizes may be
reported as bytes or MiB (2^20 bytes). Serialization span starts after the
one-way wait and includes paced reads/writes/hash overhead. Socket-call wall,
relay pacing, one-way wait, and actual request wall are separate clocks.
The recorded application queue wait is zero because there is no extra queued
frame scheduler; upstream/kernel queue residency is explicitly NOT_MEASURED.
This is application-layer emulation over physical localhost sockets, not a
remote link, measured network RTT, kernel `tc` emulation, or edge hardware.
No privileged network operation is used.

## Calibration and fail-closed gates

Before model loading, a separate bounded CPU echo process runs actual sockets:
three small dependent request/reply samples directly, then through the proxy.
Each condition has three small dependent request/replies, three 1 MiB uploads
and three 1 MiB downloads. Probe payloads are independently generated and
checked at source and sink. The two endpoint and proxy frame SHA/byte counts
must agree. Probes are never model-inference measurements. Direct loopback is
a separate control; the zero/unlimited **proxy** baseline stays in the grid.

Predeclared calibration checks:

- Round-trip increment above the proxy baseline, less nominal small-frame
  serialization, must match configured RTT within max(5 ms, 35% of RTT).
- Each limited direction's median achieved rate must be within 25% of its
  configured rate, measured across three large-frame delivery spans.
- Unlimited mode records achieved throughput without claiming an exact rate.
- During every model request, large-frame directional rates are checked against
  the same 25% tolerance. Each positive one-way wait must lie between its target
  and target + max(5 ms, 35% of one-way delay). Every frame is charged once.

A materially unrealized condition is `INVALID_IMPAIRMENT`: preserve all rows,
fail the run, stop, and suppress crossover claims. The tool does not select
favorable conditions, retry an experimental grid, widen it, or loosen tolerance.
An unsupported framing schedule fails rather than substituting synthetic delay.
Small model frames without a meaningful throughput sample are labeled
NOT_ESTIMABLE for that request and retain the independent 1 MiB calibration.

At the start of **each** condition, real NO_REUSE, COLD_MISS, WARM_HIT gate
requests must match whole FP32-serialized logits SHA and all 96 QKV SHA against
the physical same-cohort reference. Every later warmup/measured request repeats
those checks. A MISS must have 32 HIDDEN, 32 DELTAS and 32 UD computations. A HIT
must physically reuse 96 QKV, with zero data frames/bytes and zero UD math.
ES/UD/proxy ledgers, raw frame sizes, per-layer schedule, request timestamps,
separate PIDs, prepared hashes and final clean process exits are audited.

The unchanged R3-A `THRESHOLDS` and `check_parity` compare each path to fresh
Native PEFT separately. A Native FAIL remains recorded as FAIL and does not
turn a physical cache bitwise PASS into cache corruption. Conversely, a cache
bitwise failure blocks all interpretive latency claims. No thresholds change.

## Default bounded schedule

| Condition | RTT ms | ES→UD Mbps | UD→ES Mbps |
|---|---:|---:|---:|
| C00 | 0 | unlimited | unlimited |
| C01 | 0 | 200 | 200 |
| C02 | 10 | unlimited | unlimited |
| C03 | 10 | 200 | 200 |

Each condition runs six arms: NO_REUSE plus cache-enabled measured HIT
fractions 0, 0.125, 0.25, 0.5 and 1. Each arm has one warmup and eight measured
requests. Cache arms warm up with a real cold fill so that an initial HIT is
resident. Each non-HIT explicitly resets the cache and executes a real
COLD_MISS lookup/transfer/compute/insertion. Zero-HIT cache-enabled requests
therefore retain insertion overhead. A measured HIT never fakes a counter.

The zero/unlimited condition comes first. Seed 42 shuffles the other conditions,
arms and per-arm exact HIT/MISS sequence. The full realized order is saved before
execution. Arms are matched within condition rather than individually paired
randomized requests; the analysis does not claim pairing. Total default model
requests: 4 × (3 gates + 6 × (1 warmup + 8 measured)) = **228**, of which 192
are measured. Cold warmups and gate requests remain in the raw ledger.

CLI permits at most eight unique profiles, RTT 0..100 ms, each direction
unlimited or 10..10000 Mbps, 8 or 16 requests/arm, one or two warmups, and at most
five integral HIT fractions. It rejects >512 model requests, >12 GiB conservative
framed traffic allowance or >7200 s nominal resource budget. The budget uses
explicit assumed compute/startup and serialized traffic allowances; it is not
measured latency. A 7200 s model-stage deadline and 7500 s worker deadline bound
execution. One 3 h allocation includes calibration and fresh preparation.
`--profiles-json` optionally supplies the four explicit condition fields for
asymmetric rates; a zero/unlimited control is mandatory. It is exclusive with
the grid-axis arguments. No automatic sweep expansion exists.

## Timing and analysis

`request_wall_ms` is the original R3-B synchronized ES request interval. Lookup,
cached-tensor integrity validation/hash transfers, owned captures/clones,
admission/insertion, GPU forward, actual socket blocking, serialization and UD
work occur inside it. Model/tokenizer/adapter loading, fingerprints, handshake,
reset preconditions and post-wall full-logit/QKV quality export are outside and
labeled separately. CUDA event elapsed includes host-induced stream idle and
is not kernel self-time. Host/CUDA/UD/relay nested durations are never added or
subtracted to manufacture end-to-end wall or pure RTT.

The signed contrast is cache-enabled arm mean minus matched physical NO_REUSE
mean; negative favors cache. Raw ordinal contrasts, mean/median/p90 and exact
counts are retained. A predeclared 2000-resample independent-arm bootstrap
produces a small-n conditional 95% interval. An interval spanning/touching zero
is `UNRESOLVED`, including ties. Only adjacent actual measurements with opposite
CI-qualified signs can bracket a crossover. Interpolation is explicitly an
exploratory within-domain estimate, not an exactly measured root. Possible
decisions also include `ALWAYS_FASTER_IN_TESTED_DOMAIN`,
`ALWAYS_SLOWER_IN_TESTED_DOMAIN`, `NO_CROSSOVER_IN_TESTED_DOMAIN`, and
`NOT_ESTIMABLE`. Small samples, serial arm order, reused control samples and
thermal/observer effects limit inference; there is no deployment claim.

For each condition, HIT-fraction roots use the actual measured schedules.
Bandwidth roots require at least two finite measured bandwidth points;
unlimited is not a numeric endpoint. Thus the default grid cannot estimate a
finite bandwidth root. RTT roots likewise require measured opposite-sign
brackets, not multiplication of nominal RTT by assumed exchanges. Observed
32-layer dependency order is retained in source/relay/UD traces; no assertion
that wall increment must equal exactly 32×RTT is used as a gate or subtraction.

Three scopes remain distinct:

1. Resident 100% HIT request contrast excludes cold fill.
2. Controlled mixtures include actual cold lookup/insertion; reset-inclusive
   and controlled warmup/fill/reset-amortized contrasts are separate fields.
3. Real deployment lifecycle amortization is NOT_ESTIMABLE without arrival,
   refresh and residency evidence. Controlled warmups are not realistic arrivals.

R2-D is `NOT_DIRECTLY_COMPARABLE`: payload-only historical compute cohorts,
configurable element widths, possible retained h0/hL boundaries, hypothetical
RTT scheduling, and resident-reuse assumptions differ from physical bundled
FP32 deltas, ES-owned input/output, actual framing and GPU cache fill/guards.
The current model has no h0/hL socket transfers. Optional `--r2d-dir` validates
the five exported R2-D files and SHA map read-only, records their assumptions,
and never pools old timings or requires inaccessible original source artifacts.
A full-HIT advantage is insufficient to validate an R2-D network root.

## Running and artifacts

On SERAPH, after manually transferring your reviewed source changes and using
branch `exp/raw-semcache-r3`, create logs before Slurm opens its log paths:

```bash
cd /data/khuss/repos/GP_semcache
mkdir -p logs
bash -n jobs/r3c_network_one_shot.sh
sbatch jobs/r3c_network_one_shot.sh
```

The launcher activates conda `semcache` (optional `R3C_CONDA_ROOT`), selects one
GPU/four CPUs/64 GiB on `batch_eebme_ugrad` without a node pin, sets offline HF
cache `/data/khuss/huggingface`, logs branch/HEAD/dirty state, checks dependencies
and cached OPT/tokenizer/bin-shard presence, and runs fresh immutable original
95 prepare plus persistent ES/proxy/UD and read-only reanalysis. No pilot,
96/97 forensics, precision sweep, downloads or privileged networking runs.

Supported main invocation within the allocation:

```bash
python -u scripts/100_run_r3c_network_one_shot.py \
  --phase run --enable-gpu-experiment --device cuda:0 \
  --rtt-ms 0,10 --bandwidth-mbps unlimited,200 \
  --hit-fractions 0,0.125,0.25,0.5,1 \
  --requests-per-arm 8 --warmup-runs 1 --seed 42 \
  --output-dir results/r3/c/network_one_shot_<jobid>_<UTC_timestamp>

python scripts/100_run_r3c_network_one_shot.py --phase analyze \
  --inputs results/r3/c/network_one_shot_<jobid>_<UTC_timestamp> \
  --output-dir results/r3/c/network_one_shot_<jobid>_<UTC_timestamp>_analysis
```

Outputs must be new/empty, under `results/r3/c/network_one_shot_*`, and outside
the input cohort. Read-only analysis rejects missing/tampered files, changed
source, unclean/failed runs and incomplete schedules. Negative scientific
results and Native FAIL still allow a properly completed diagnostic.

| Artifact | Evidence |
|---|---|
| `r3c_one_shot_results.json` | execution, source HEAD/SHA maps, immutable input hashes, GPU identity, gates, output hashes, reason/traceback |
| `r3c_one_shot_summary.txt` | concise scope, execution, Native result and crossover decisions |
| `r3c_requests.jsonl` | all gates/warmups/measurements, exact order, 96 QKV SHA, logits SHA, Native metrics, timing/counters/profile/calibration/preconditions |
| `r3c_network_calibration.json` | all requested/realized directions, raw probes, tolerances, residuals, validity, proxy overhead |
| `r3c_break_even.csv` | signed raw/statistical arm contrasts, byte savings, uncertainty and controlled amortization |
| `r3c_break_even_decision.json` | measured brackets, unresolved/no-root states, invariant qualification flags |
| `prepare/`, `prepare_process/` | original 95 immutable reference, matrices, preflight/GPU UUID, worker stdout/stderr/lifecycle |
| `calibration/` | source/echo/proxy probe reconciliation, raw probes, echo stdout/stderr and exit records |
| `proxy/` | original frame SHA/bytes, monotonic delivery/pacing/wait traces, profile IDs, stdout/stderr, stopped-thread/lifecycle evidence |
| `es/` | actual original ES/UD wire traces, per-layer/UD timing, original R3-B raw rows, cache events, condition gates, preconditions, source physical reference, stdout/stderr and descendant cleanup |

Process failures retain partial rows/traces and concrete reasons, set FAILED
with NOT_ESTIMABLE analysis, stop the finite grid and clean listeners/children.
No incomplete conditions are silently omitted from a successful analysis.
The launcher records failures and post-validates artifacts with the same CLI.

## Local verification and remaining execution

CPU verification needs no Torch, Transformers, PEFT, model weights or downloads:

```bash
PYTHONPATH=src python3 -m unittest tests.test_r3c_analysis \
  tests.test_r3c_proxy tests.test_r3c_processes -v
python3 -m compileall -q scripts/100_run_r3c_network_one_shot.py \
  src/semcache/edgelora/r3c_*.py tests/test_r3c_*.py
python3 scripts/100_run_r3c_network_one_shot.py --help
bash -n jobs/r3c_network_one_shot.sh
```

Socket tests need permission to create/bind loopback TCP sockets; a sandbox
denial is not a protocol PASS. CPU fixture mode is explicitly
`--phase fixture --test-fixture-only`; calibration-only and no-socket dry-run
are `--phase calibrate` and `--phase dry-run`. CPU fixture data are always
TEST_FIXTURE_ONLY and forbidden as GPU evidence. The opt-in GPU unit test needs
`SEMCACHE_R3C_GPU=1`, a unique `SEMCACHE_R3C_OUTPUT`, dependencies, model cache and
a real Slurm GPU allocation; it is skipped locally. GPU findings are NOT_RUN
on the personal Ubuntu implementation machine. No SERAPH filesystem, Slurm
submission or remote mutation is performed by the implementation task.

For interpretation, share the root results/summary, calibration, break-even
CSV/decision and failure stderr if any. Raw requests and per-process traces are
needed to audit a specific contrast or condition failure. R3-C failure or no
crossover limits this physical endpoint/controlled workload experiment; it
does not retroactively invalidate M1–M9 or C1.5–C9.

Implementation-machine verification (2026-10-30, no model dependencies):
29 new tests: 28 PASS, 0 FAIL, 1 GPU SKIP. The 190 selected frozen R3-A/B,
precision/forensics, R2-D, M9-A and R2-C4 regressions: 173 PASS, 0 FAIL,
17 dependency/GPU SKIP. Compilation, help and Bash syntax checks passed.
The final default 4-condition/5-fraction CPU fixture completed all 228 requests,
passed the independent ES/UD/relay audit and read-only reanalysis, and recorded
TEST_FIXTURE_ONLY_NO_GPU. This verifies scheduling/transport/cache plumbing,
not OPT numerics, GPU tensor ownership or measured GPU crossover.

An earlier CPU smoke stopped at 221 requests because the recorded wait used a
later helper-entry origin than its deadline. Its failing frame recorded
4.9768 ms whereas header-ready to serialization start spanned 5.1279 ms.
The report remains FAILED and unchanged. The fix measures delay from the exact
deadline origin and has a dedicated regression test; no tolerance was relaxed.
Initial pacing development likewise retained a failed CPU calibration, then
fixed repeated wakeup-overhead charging with the documented bounded credit.
Local verification records are under
`results/r3/c/network_one_shot_local_verification_20261030T061847Z/`.
All 535 preexisting tracked file hashes, including frozen thresholds and
95–99/production sources, remained unchanged. Real OPT/GPU findings: **NOT_RUN**.
