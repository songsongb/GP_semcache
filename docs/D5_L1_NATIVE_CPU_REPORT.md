# Native Light follow-up — PARTIAL_BLOCKED

## Source and execution boundary

Worktree: `/home/songsong/GP_semcache_lightperf`; branch:
`exp/d5-l1-l5-lightperf`; starting/current HEAD:
`6708d369f42577a3fbe8b46411aa98cdbac4b776`. The earlier Light files were already
committed in this HEAD and were used as the implementation baseline. Strict
baseline `4916836ce328e119f3a8d85c6c1afc7cf8d363e5` is retained in its history.
Follow-up edits are uncommitted and tracked-dirty. No commit/push/branch operation,
remote operation, package installation, download, external API, GPU inference or
Slurm query/submission/cancellation occurred. Only local TCP tests required the
approved socket-capable execution context.

`git diff` against the Strict baseline, intersected with paths present in that
baseline, reports **zero existing Strict paths changed**. Original source/CLI/job/
auditor/tests are unchanged. Prior Light development receipts remain sealed and
are not reused as approval. Historical GPU evidence is absent locally, so a full
historical tensor/manifest re-audit cannot be claimed.

## Implemented worker

Implementation status: **IMPLEMENTED_NOT_NATIVE_RUNTIME_VERIFIED**.

`r5_d5_lightperf_runtime.py` implements the independently guarded coordinator,
native ES worker, isolated UD worker, shared serial request engine and shared CPU
UD session. Existing prepared base/tokenizer and matrix/registry validators are
called directly. The ES uses the existing bare OPT setup, one PhysicalOwner/
SemanticIndex per adapter and native full prefill/decode. Each UD is a separate
Python process with CUDA/HIP hidden, no transformers/full model, actual FP32 CPU
LoRA matrices and the frozen sparse protocol.

Unlimited/200/100/50 Mbps shaping and factor1/.5/.25 measured UD service emulation
are wired into those actual channels/compute calls. There is no Strict worker
fallback, fixture CLI, reduced LoRA arithmetic or synthetic production output.
Each condition is one uninterrupted stream with persistent per-adapter cache/UD;
only the final adapter completion ACK triggers STOP/BYE before local admission.
Original 5/20/50 selection/routing/group and continuous-identity contracts remain.
Failed resource estimates remain blocked; extended jobs require a separate opt-in.

Original safety/calculation primitives remain unmodified:

* `PhysicalOwner`, per-layer EXACT filters, tensor proofs, owned captures and
  authenticated consumed snapshots from `RequestValidation`.
* Original sparse gather/project/scatter, bare Linear correction, full native
  attention/FFN/LM head and target-owned decode past KV.
* Actual `StreamingImpact` FP64 attention reduction, F/A/I/S, CHU, admission,
  eviction and PBR. No zero-impact substitute or proof removal.
* Session/adapter/shape/sequence/hash validation and generation completion ACK.

Static call preservation and scalar fixtures support this claim about code wiring;
they do **not** establish native numerical fidelity. That requires the tests below
on SERAPH CPU and later separately approved matched five-request GPU evidence.

## Changed and added files

Added:

* `src/semcache/edgelora/r5_d5_lightperf_runtime.py`: coordinator/ES/UD/session engine,
  actual resources, source/approval guards, deadline and bounded child cleanup.
* `src/semcache/edgelora/r5_d5_lightperf_receipts.py`: compact request/profile/wire/
  policy summaries and endpoint byte/frame-chain equality.
* `tests/test_r5_d5_lightperf_native.py`: five mandatory native Tiny OPT tests;
  missing dependencies fail instead of skipping.
* `tests/test_r5_d5_lightperf_worker.py`: fifteen stdlib orchestration/approval/
  lifecycle/failure contracts, including actual local TCP.
* This report.

Updated Light-only files:

* `r5_d5_lightperf.py`: engine V2 plans, source/input/version/semantic asset pinning,
  native zero-skip CPU approval and run/_worker/_ud CLI dispatch.
* `r5_d5_lightperf_timing.py`: CPU-only bounded in-memory logit observer, request
  deadlines, semantic/cluster metrics and existing impact metadata.
* `r5_d5_lightperf_emulation.py`: condition deadlines, per-send pacing metrics,
  expected request stream validation, LoRA/serialization/hash service summaries.
* `r5_d5_lightperf_io.py`: explicit engine identity and measured summary IO costs.
* `r5_d5_lightperf_analysis.py`: V2 approval/version binding, external UD seals,
  primary/final receipt equality and per-request wire/LoRA checks. Comparable GPU
  identity uses UUID/name/device/capacity; transient free-memory differences are
  retained metadata rather than falsely treated as different devices.
* `r5_d5_lightperf_workload.py`: endpoint file-count forecast now includes both ES
  and UD receipts/failure reserves; selection/limits are unchanged.
* `tests/test_r5_d5_lightperf.py`: V2 receipt/launch gates and stronger test fixtures.
* `jobs/r5_d5_lightperf.sh`: manual one-condition baseline-five template with
  mandatory fresh CPU inspect, source/GPU UUID/allocation and explicit opt-ins.
* Existing design/runbook updated; the old CPU report is marked historical.

The script `scripts/159_r5_d5_lightperf.py` is reused unchanged; dispatch changes
are in its imported Light module. No original Strict source lines changed.

## Timing and receipt semantics

Each primary request receipt contains identity/order, output IDs/text/hashes,
quality/raw scoring version, physical HIT and skipped/fresh rows, A/B/C/D source
categories, occupancy/entry/policy counts, original integrity/admission metrics,
actual impact hash and phase/component wall scopes. Semantic encoding and cluster
assignment are separately measured. Profiles retain inclusive and observed
exclusive scopes after collapsing layer/role dimensions; they are never summed
as a reconstruction of E2E.

UD receipts record measured exact LoRA compute, emulated service/pacing time,
requested/achieved factor, projection rows, serialization/deserialization/hash
times and ordered completions. Both directions retain actual encoded frame bytes
and chained frame hashes; ES/UD summaries must agree. Requested link rate and
achieved Mbps are recorded separately; achieved spans include inter-frame compute
and wait gaps and are not a geographic/network hardware measurement.

Application arrival→first host-visible token and arrival→primary receipt written
are recorded as TTFT/E2E. Quality and primary persistence are measured separately
inside E2E. Initialization and final run.json/sealing are excluded. Model prefill/
decode scopes are wall wrappers, not kernel durations. Exact QKV kernel time,
driver version, network queueing, total profiler/observer overhead and final seal
duration are **NOT_MEASURED**. No new tiny-scope CUDA synchronize was introduced.
Safety serialization/hash/clone work remains present and profiled.

Outputs: one flat condition directory plus fresh `RUN__ud_Adapter_A/B/C` sibling
directories. Each is independently sealed and bound by run.json inventories.
Primary bodies are write-once; final application timing is recorded in run.json,
with primary body hashes. Any failure.json, failed child, wrong request/order/
counts/hash, invalid ACK/BYE, changed source or invalid CPU audit rejects success.
The coordinator watchdog terminates only its own process group; no Slurm job is
queried/cancelled. Failure finalization preserves the original exception.

## CPU tests actually run

All commands used CUDA_VISIBLE_DEVICES='' and PYTHONPATH=src:tests.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest \
  test_r5_d5_lightperf test_r5_d5_lightperf_worker -v
```

Final: **41 PASS, 0 FAIL, 0 ERROR, 0 SKIP; 1.033s** unittest wall time.
An earlier 39-test sandbox attempt had four socket PermissionErrors (0.661s);
the identical command in the approved loopback context passed 39/39 (0.855s).
After two approval/job tests were added, 41/41 passed (0.793s), then 41/41 passed
again after moving infeasible-resource rejection before approval tests (1.033s).
No skipped test was converted to PASS.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest test_r5_d5_lightperf_native -v
```

Final: **0 PASS, 5 FAIL, 0 ERROR, 0 SKIP; 0.003s**. All five fail at the explicit
dependency gate: torch, transformers, numpy unavailable. No Tiny OPT forward or
native LoRA test ran. Earlier runs of the four-test version failed for the same
reason (0.002s and 0.014s), before the native deadline-abort case was added.

Required native coverage, implemented but **NOT_VERIFIED** here:

1. FULL/EXACT/APPROX Strict↔Light full-vocabulary logits in RAM, generated tokens,
   actual attention impact, HIT/MISS/rows, cache policy/eviction/PBR, routing and
   final adapter ACK/STOP/BYE through the real serial engine and two spawned UDs.
2. Actual native generation under all initial rate/factor choices, comparing
   logits and row decisions to baseline with identical arithmetic.
3. Native cached tensor `.data` corruption, expired request scope and no fallback.
4. Native deadline failure without admission/retry.
5. Native greedy EOS termination compared with Strict.

The memory-only observer is restricted to CPU TEST_FIXTURE_ONLY Tiny OPT with
≤4 layers, ≤256 vocabulary, ≤64 input and ≤8 generated tokens; no GPU/CLI raw
logit export exists. Synthetic native test requests deliberately cover reuse;
production NATURAL plans cannot use these fabricated samples/routing.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest \
  test_r5_natural_shutdown test_r5_natural_phase3a \
  test_r5_subsequence_generation test_r5_subsequence_validation \
  test_r5_sparse_physical test_r5_subsequence_audit \
  test_r5_natural_approx test_r5_subsequence_natural -v
```

**102 PASS, 0 FAIL, 0 ERROR, 8 SKIP /110 run; 2.728s.** The eight existing native
cases skip because their frozen dependency decorators require torch/transformers.
This is not zero-skip approval. New native tests do not share that skip behavior.
Python compilation, bash syntax and git diff whitespace checks pass.

## Assets, resources and approval status

Local package discovery: torch/transformers/numpy/rouge all absent. Real prepared
workload/multi/base manifests found in this worktree: **0**. SERAPH original path
`/data/khuss/repos/GP_semcache` is absent here. No real dataset request IDs, model
weights, tokenizer, TinyBERT, LoRA matrices or frozen Strict GPU receipts were
validated. No CPU approval file for actual GPU launch was issued.

Original 5/20/50 resource limits remain. At 512 input/original generation caps,
the unmeasured 5s/forward+600s reserve assumption gives five-request Qasper/2Wiki
1400s and MultiNews 3000s per condition. Qasper/2Wiki50 forecast8600s, MultiNews20
10200s and MultiNews50 24600s remain BLOCKED_INFEASIBLE at7200s. These are static
forecasts, not measured runtime. Do not lower generation or increase limits to pass.

Future manual allocation requires one compatible GPU/actual UUID, ≥12GiB free
after load, <22GiB peak, 4 CPUs, 64GiB allocated RAM/≥32GiB available and ≥1GiB
disk. Remaining allocation time is supplied explicitly, not queried from Slurm.
One condition is a single continuous session; no job splitting or restoration.

**L2 five-request readiness: PARTIAL_BLOCKED.** Remaining blockers:

* Commit/review the new tracked and untracked source files; current dirty HEAD
  cannot be used to prepare a clean production plan.
* On SERAPH's existing offline native environment, pass the five native tests and
  all required regressions with zero failures/errors/skips. Native code may still
  require fixes exposed by that first real execution; none are claimed resolved.
* Supply and reconstruct actual prepared assets and dataset-specific first-five
  lists; obtain fresh PLAN + native CPU audit for the new committed source.
* Only then obtain separate user approval for a bounded five-request GPU trial;
  actual GPU resources/backend/fidelity remain GPU_NOT_RUN/NOT_VERIFIED.

The updated `D5_L1_L5_GPU_RUNBOOK.md` includes all three dataset-specific prepare,
CPU validation/inspect commands, later manual FULL/EXACT/APPROX invocations and
expected resources. Template fails closed without approval; it submits nothing.
20/50 and L3–L5 remain separate later approvals; L5 plot/series work is not part of
this native-worker completion task. No Cartesian campaign was run.

Historical Strict MultiNews fidelity is not used as native Light PASS evidence.
Persisted full-vocabulary independent audit stays NOT_EVALUATED and
correctness_qualified_speedup stays NOT_ESTABLISHED. No GPU speedup, correctness/
safety qualification or 50-request success is claimed.
