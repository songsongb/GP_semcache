# R5-I3P measurement correction

Implementation map, before changes: retain the existing eager OPT forward,
CPU UD mathematics, semantic/controller/cache identity and eligibility. Add a
scoped timing driver around the unchanged C1 forward; add opt-in generation
timing boundaries; measure the legacy budget checker before replacing it. Then
add writer-accounted ES/UD evidence, fresh-process A/B blocks, independent replay
and block-level statistics. Default R5 and frozen R4 behavior remain available.

Job 152301 is user-reported verified physical integration evidence on SERAPH. Its requested path
`results/r5/opt_152301_20261031T074321Z/` is absent locally at initial inspection.
The reported 115.44s versus 257.14s (55.1%) contrast is **confounded**, never a
qualified cache speedup. Missing original artifacts prevent local independent
attribution of that run. New CPU filesystem/fixture measurements are separate.

Inspection identifies three distinct unmeasured regions: recursive Evidence.check
and input-KV inspection before C1's internal host timer, post-forward token/state
handling and synchronization, and JSON appends after token availability. Their
contributions must be measured, not assigned to one suspected cause.

## Measured attribution and historical scope

The original GPU records are on SERAPH at
`/data/khuss/repos/GP_semcache/results/r5/opt_152301_20261031T074321Z/`.
They are deliberately neither fetched nor modified by this task. Historical
analysis is `HISTORICAL_GPU_EVIDENCE_UNAVAILABLE_LOCALLY`, not implementation
failure. Transfer/read at minimum `es/r5_tokens.jsonl`,
`r5_generation_outputs.jsonl`, `es/es_plan.json`, `r5_identity.json`,
`r5_requests.jsonl`, `r5_latency_summary.json` and `r5_output_manifest.json`.
Retain `es/es_wire.jsonl`, `es/ud_wire.jsonl` and `es/ud_layers.jsonl` for traffic
and CPU UD attribution. Even those original records do not directly time the
budget checker: historical wall-minus-host residual alone cannot establish cause.

Before implementing incremental accounting, the **actual unchanged legacy
Evidence.check** was measured on controlled CPU filesystem trees (three checks
per size). Saved evidence: `results/r5/performance/local_verification_20261031/legacy_scan_attribution.json`.

| Existing files | Median budget-check ms |
|---:|---:|
| 0 | 0.050 |
| 1,000 | 12.720 |
| 8,000 | 121.866 |
| 32,000 | 740.643 |

This confirms a file-count-dependent contribution **on this local filesystem**.
It does not confirm that scanning caused Job 152301's approximately 10–22 second
step-wall discrepancies. Neither the historical GPU slowdown nor its elimination
has been reproduced locally. Filesystem/backend differences and other preflight,
KV inspection, synchronization, serialization and instrumentation costs remain
separate possible contributors. The diagnostic job measures those directly.
The controlled tree uses Python's temporary directory (TMPDIR); its filesystem
may differ from the experiment output filesystem. Its numbers must not be used
as estimates of SERAPH NFS cost. The diagnostic job's `--legacy-scan` timings
measure the actual growing block output directory on the serving filesystem.

`ProfiledDriver` invokes the original `QualityDriver.c_forward`; it contains no
copy of the OPT forward. A scoped line tracer instruments only that wrapper,
partitioning budget check, preflight/input-KV inspection, distributed forward,
past inspection, logit extraction, evidence/validation and completion append.
Passive wrappers time the original actual model call, explicit synchronization,
logits and evidence writes. The existing R5 generation loop gains opt-in spans
for selection, output state, post-forward work, token JSON append, completion
control and admission. Its default has no profiler and retains the original
mathematics, control order, eligibility and cache representation.

Per-step reports preserve exclusive wrapper bins **and** nested model/evidence/IO
spans. Never add nested values to their parents. `wall_to_availability_ms` excludes
the subsequent token JSON append, while `step_through_json_ms` includes it; the
profiling record's own final append is outside its self-reported duration and
inside application wall time. `profiler_finalize_ms` identifies wrapper report
assembly. Tracer local callback cost is measured, not claimed to represent all
profiler overhead. Both arms use identical instrumentation; no uninstrumented
performance claim is made.

Timing includes request preparation; encoder/cluster/candidate/eligibility/KV
lookup from actual R5 discovery; budget checking; actual distributed forward;
input and output KV inspection; evidence capture; logits; greedy selection;
output/past handling; synchronization; JSON and tensor/logit writing; completion;
admission/eviction. Each step records request/step index, observed file/byte counts,
scan count and cumulative scan time, CUDA event elapsed time and GPU memory when
available, real UD compute/serialization receipt time and Channel send/receive
elapsed time. CUDA events include stream/host/socket idle; they are not kernel
self-time. TCP timings include original Channel framing/serialization work.

## Evidence accounting and integrity

`Writer` is a shared SQLite byte/hash ledger used by ES, UD and parent pipe
drains. Every data/blob/metadata append or replacement obtains BEGIN IMMEDIATE,
checks the global quota before writing, writes exact bytes, records size/digest,
and updates constant-size total/file counters. Append hashes are incremental.
Large tensor/logit writes and both processes' JSON/control logs are counted;
stdout/stderr are drained through the same quota instead of unbounded files.
There is no assumption that an ES-local counter sees UD growth.

SQLite uses DELETE journaling, FULL synchronization and an 8,192-page bound.
A conservative 64MiB reservation covers its <=32MiB database plus rollback journal.
Data bytes are exact, with actual accounting-file bytes and total observed bytes
reported separately. A block budget must exceed the accounting reservation and
the original 16MiB forward headroom. Normal checks read shared counters and stat
only known accounting files; they do not walk the growing data tree.

Default reconciliation interval is 16 evidence checks (forward-entry and
request-end checks, not exactly 16 Decode tokens). Each reconciliation obtains
the same writer lock, independently inventories and hashes the full data tree,
and rejects additional/missing/corrupt files, symlinks, mismatched totals and
accounting namespace changes. Unknown external writers are detected at that
bounded checkpoint; the code does not claim to prevent arbitrary third-party
disk writes between reconciliations. All managed writers reserve bytes first.
Final reconciliations occur after ES/UD/pipe writers have exited, before success;
the independent read-only auditor repeats inventory/hash/quota checks separately.

Overflow or mismatch permanently marks the ledger failed. Existing files and
partial ledgers remain; the outer launcher preserves failure stage/cause even
when a failed block ledger refuses more writes. The top output manifest hashes
all artifacts, including finalized ledgers. Reanalysis opens ledgers read-only.
This provides process-failure detection and evidence preservation, not a promise
of lossless recovery from power loss halfway through a data-file write.

Profiles are `STRICT_AUDIT` (all existing detailed tensors and live checks) and
`SCALABLE_PERFORMANCE` (identical inference/live checks, tensor digests and full
next-token logits, without unnecessary tensor files). Compact digests are not
full strict tensor evidence. The old R5 strict/scalable paths and old recursive
budget strategy remain available through their unchanged default entrypoint.
`--legacy-scan` adds the actual recursive checker traversal to the new pipeline
for diagnosis, while retaining the shared quota. It measures traversal cost;
it is not a byte-for-byte recreation of the historical execution environment.

Only process-local IO dependencies are injected into frozen modules; their source,
OPT forward and CPU LoRA operations remain unchanged. Injection is opt-in and
restored on exit. The inference is still semantic → cluster/index → automatic
decision → owned Raw-QKV → real ES/UD TCP → prefill/Decode/final generation.

## Balanced design, isolation and warmup

A is distributed no-reuse, including semantic overhead. B is the actual global
Raw-QKV path. One repetition is ABBA; subsequent repetitions alternate BAAB/ABBA.
Explicit `--start-order BAAB` is supported. Seeds/order manifests are deterministic;
the seed controls bootstrap resampling, not a claim of randomized serving order.
Adapter/model preparation retains the fixed seed42 research identity.

Every block has a new ES process, new model load, new encoder, new cache/index,
new clusterer, new counters and an independent output namespace. Both conditions
execute the same original ten controlled texts in their original order, normally
tokenized, with identical generation/evidence/configuration. No expected labels
enter dispatch. Each B block must independently realize four HITs, four admissions
and one capacity eviction. A must realize no HITs/admissions/evictions.

After model setup, identical first-request warmup runs in no-reuse mode and a
separate UD session, with no admissions. That UD exits. Measured serving starts
with a fresh UD PID/session and newly constructed empty ForwardOrder; its actual
grants/parents/finished/layer/current state is captured at construction. Cache,
index, cluster centroids/counts, request IDs and driver cache type are fresh.
No request KV survives the generation call. Warmup uses the same base model and
encoder object in the ES but no serving metadata survives; weights remain eval
and fingerprinted. Measured UD BLAS state is cold by design in both arms; this
is not a claim that every possible tensor shape is fully warmed.

Setup and warmup costs are separate from serving. Warmup requests cannot appear
in measured requests/HIT counts. Every new block's base/tokenizer/adapter/encoder
fingerprints and GPU UUID/name must match; volatile GPU memory counters may differ.
Separate PID/session namespaces prove no cross-block UD grants/history remain.
Per-block initial/final state and chronological timestamps are independently audited.

## Statistics and qualification

Reports separate physical HITs, non-HIT negative controls, all requests, prefill,
Decode, TTFT, model E2E and application E2E. They include payload/wire traffic,
source admission/eviction, profiler and scan/write overhead, CPU UD/TCP time,
memory, warmup/setup and fingerprints. Source-generation subtotals are identified;
actual workload E2E already includes source/admission costs and must not add them
again. Admission cost per realized HIT is a descriptive amortized allocation.

Matched A/B requests must have identical text, exact tokens, mask, positions,
policy, user/adapter identity and physical namespace. Every non-HIT comparison
checks full prefill projection rows and tensor payload equality; header/control
bytes can differ and are reported. Logits/text/tokens, argmax, max_abs, relative-L2,
KL and bitwise logit equality are separately retained. Equal-history steps alone
are valid direct numerical comparisons. The strict per-block auditor checks
source/prefix/Decode bytes; the performance report does not establish new full
cross-condition tensor or Native PEFT parity.

The principal bootstrap unit is the whole independently initialized **four-block
repetition**, retaining both adjacent AB/BA contrasts and all ten dependent
requests. It is never the ten request rows. Reports give means, medians, sample
standard deviations, reproducibly seeded paired percentile 95% intervals,
order-specific contrasts, per-request contrasts and total workload contrasts.
One cycle has no inferential CI. Fewer than eight independent cycles are labeled
exploratory; eight is a conservative design minimum, not a significance guarantee.
Fresh processes alone do not prove independence of hardware/time effects.

Mandatory non-HIT drift checks compare actual distributed-forward time and
admission/lookup-adjusted E2E. A cycle-bootstrap interval excluding zero flags
systematic residual drift without an invented millisecond threshold. Detailed
semantic, serialization, evidence and chronology observations remain available
to investigate that residual; it is never attributed to prefix reuse. Descriptive
slopes/correlations versus chronology and accumulated evidence files (globally
and within each block) are associations, not causal proof.

Functional gates cover timing attribution, evidence accounting, isolation, balanced
order and independent audit. GPU execution is separate. Statistical reduction
flags require genuine GPU execution, sufficient independent cycles, compatible
outputs/settings and no flagged residual MISS drift. The logit-tolerance checks
reuse **unchanged R3-A thresholds** for every aligned-history step, including
allclose violations, maximum absolute error, relative L2, KL and argmax. No new
numerical tolerance is invented. `R5_I3P_PERFORMANCE_QUALIFIED` can qualify only
this instrumented distributed comparison after genuine GPU audit, at least eight
cycles, exact token agreement, passing frozen logit tolerances, no flagged MISS
drift and corrected accounting (no legacy traversal mode). Qualification does
not require a positive effect; statistical reduction flags additionally require
a 95% interval below zero. Native PEFT strict parity, full strict numerical
qualification and correctness-qualified speedup remain separate and are never
inferred. A functional PASS is not a performance PASS. Bootstrap inference is
conditional on cycles being exchangeable; process isolation/order balance alone
cannot prove that hardware/time effects are independent.

## Commands and SERAPH preflight

```bash
python3 scripts/112_run_r5_balanced_benchmark.py --phase diagnose \
  --output-dir results/r5/performance/my_unique_local_scan
python3 scripts/112_run_r5_balanced_benchmark.py --phase fixture --test-fixture-only \
  --repetitions 1 --start-order ABBA --generation-budget 2 --warmup-requests 1 \
  --output-dir results/r5/performance/my_unique_fixture
python3 scripts/113_analyze_r5_balanced_benchmark.py \
  --inputs results/r5/performance/my_unique_fixture \
  --output-dir results/r5/performance/my_unique_analysis
PYTHONPATH=src:tests python3 -m unittest tests.test_r5_contracts tests.test_r5_semantic \
  tests.test_r5_controller tests.test_r5_processes tests.test_r5_perf_evidence \
  tests.test_r5_perf_schedule tests.test_r5_perf_processes -v
bash -n jobs/r5_opt_balanced_benchmark.sh
```

CPU fixtures require localhost socket permission. Help/dry-run load no GPU or
model. Output directories are immutable/non-overwriting under
`results/r5/performance/`; incomplete runs preserve partial artifacts and false
qualification gates. Read-only analysis refuses incomplete order, missing data,
inventory corruption, state leakage, profile/config mismatches or false HIT work.
Independent full replay also requires the small R3-A prepared artifact directory
at the base_prepared path recorded in each plan. When transferring runs, include
that directory and preserve/mount the recorded path, or run analysis on SERAPH
before transfer. Analysis reads its manifests, adapter and reference files; it
does not load the OPT checkpoint or launch GPU inference.

On SERAPH: commit every execution source on exp/raw-semcache-r5-integration;
create logs/ before submission; activate semcache with compatible Torch, NumPy,
Transformers and PEFT; retain offline OPT/model tokenizer and genuine TinyBERT at
**34707a33cd59a94ecde241ac209bf35103691b43** under /data/khuss/huggingface.
No download is attempted. The job requests exactly one moana-y4 RTX3090, 4 CPU
cores, 64GB and 3 hours. It checks allocation/offline dependencies/disk, prepares
fresh base and verified adapter matrices for that GPU UUID, and refuses an old
GPU identity. Runtime also checks the committed source, fingerprints and opt-ins.

The default 1GiB block cap includes the conservative ledger reservation; preflight
requires worst-case block budgets plus setup/report headroom. Diagnostic mode
reserves >10GiB free space for eight strict blocks. Performance mode with eight
cycles reserves >34GiB for 32 blocks; real compact storage should be much smaller,
but that is not presumed. Increase per-block budgets explicitly if strict data
requires more space; do not silently downgrade evidence. Setup/reload cost and
the three-hour allocation constrain feasible repetitions. No unsupported sample
size guarantee is made.

```bash
# Operator-controlled diagnostic: strict ABBA with legacy scans, then strict
# ABBA with incremental checks. Compare directly timed check costs, not an
# unbalanced cross-run application speedup claim.
R5_ENABLE_GPU=1 R5P_ENABLE_GPU=1 R5P_MODE=diagnostic \
  R5P_SOURCE_COMMIT=$(git rev-parse HEAD) sbatch jobs/r5_opt_balanced_benchmark.sh

# Separate lower-overhead balanced evaluation: 8 cycles / 32 fresh blocks, G8.
R5_ENABLE_GPU=1 R5P_ENABLE_GPU=1 R5P_MODE=performance R5P_REPETITIONS=8 \
  R5P_SOURCE_COMMIT=$(git rev-parse HEAD) sbatch jobs/r5_opt_balanced_benchmark.sh
```

These commands are prepared only, never submitted by this task. Diagnostic and
performance profiles are distinct experiments, not interchangeable timing samples.
Historical GPU integration correctness is preserved; **new balanced GPU execution
and statistically supported GPU improvements are NOT_RUN locally**.

## I4 handoff (not implemented)

Start I4 only after SERAPH diagnostic/accounting/isolation/audit gates pass and
remaining performance drift is explained. Expansion order: compatible shared
adapter logical-user scale; independently prepared/verified multiple LoRA
adapters; adapter-aware isolation and explicitly permitted sharing; measured
Frequency/Recency policy; Semantic Impact only with actual attention observations;
concurrency only after sequential multi-user correctness. No I4 feature, new
semantic policy, model, cache format, eligibility relaxation or LoRA mathematics
is introduced here.

## Final local verification and changed files

Evidence root: `results/r5/performance/local_verification_20261031/`.
Machine-readable final record: `verification_locked.json`.

| Validation | Actual result |
|---|---|
| Existing dedicated R5 tests | 36 passed, no failures/errors/skips |
| Final performance-specific tests | 18 passed plus1 late-failure guard test; no failures/errors/skips |
| Total unique dedicated R5 tests | **55 passed** |
| Relevant unchanged R4 suites | 60 run:55 passed,4 dependency skips,1 historical branch-gate failure |
| Scalable ABBA | 4 fresh blocks,40 measured requests,80 measured forwards,8 HITs,8 admissions,2 evictions |
| Strict BAAB | Same measured counts; strict physical tensor/KV audit PASS |
| Warmup, each experiment | 4 separate requests /8 forwards, excluded from measured counts |
| Independent repeated analysis | All10 deterministic reports byte-identical in two analyses and source scalable reports |
| Cross-profile numerical control | All40 request results match tokens, text and serialized logits between strict/scalable runs |
| Compilation /Slurm shell syntax /diff checks | PASS |
| GPU benchmarks /Slurm submissions /remote Git operations | **0 /0 /0** |

Logs: `r5_all_tests_initial.log` contains all36 existing R5 tests plus17 initial
performance tests; `performance_tests_locked.log` contains18 final performance
tests (including actual TinyBackend math compatibility and concurrent snapshot
checks). `late_failure_guard_test.log` adds the post-audit failure case: any late
source/budget/report failure revokes all gates. No failing test was relabeled.
`frozen_r4_regressions.log`
preserves the known R4-A test expecting MISSING_DEPENDENCY after its frozen
branch gate rejects this R5 branch. Its source is unchanged. The four skips are
missing datasets, official ROUGE and two actual-OPT/HF dependency checks. This
host has no Torch/NumPy/Transformers/PEFT runtime, so no GPU/real-model timing is
verified. The sandbox-denied initial TCP attempt is preserved as failure evidence.

Saved artifacts: final `scalable_abba_locked/`, `locked_analysis_a/`,
`locked_analysis_b/`, `strict_baab_fixture/`, `strict_baab_analysis/`,
`analysis_reproducibility.json`, `statistics_and_profile_validation.json` and
`historical_preservation.json`. The strict data were independently reanalyzed
after final statistical/audit hardening. Physical forward/generation/controller
sources match the final scalable run. After strict capture, Writer.status was
hardened to snapshot counters and ephemeral journal metadata under one shared
transaction, preventing an ES/UD commit race; inference and blob-writing semantics
were unchanged. Its concurrent test and full scalable ABBA rerun pass. A final
launcher-only failure guard was then added and tested without altering the
successful inference path. These source variants remain recorded; no original
artifact is rewritten. Each benchmark has its own content-hashed output manifest.

For the strict CPU BAAB fixture, ordinary budget-check medians were
0.393–0.434ms. A real periodic full reconciliation cost244–271ms per block:
it remains counted, not hidden. Each block made30 budget checks and one serving
reconciliation; final parent reconciliations followed process exit. The observed
input-KV/preflight medians were5.76–7.12ms and post-forward medians0.118–0.152ms.
These observations show distinct contributors; they are not GPU latency or proof
of the historical root cause. The local fixture did **not** reproduce the old
10–22s residual pattern. One cycle has no inferential interval; all GPU,
statistical-reduction and performance-qualification flags remain false locally.

The58,524-file pre-task snapshot shows only the two intended R5 source edits;
all historical artifacts and frozen R3/R4 sources are unchanged.

| Added file(s) | Purpose |
|---|---|
| src/semcache/edgelora/r5_perf_timing.py | Frozen-forward boundary profiler and actual scan microbenchmark |
| src/semcache/edgelora/r5_perf_evidence.py | Shared writer ledger, bounded quota, reconciliation and accounted pipes |
| src/semcache/edgelora/r5_perf_schedule.py | ABBA/BAAB manifests, block/pair/repetition identities, pinned encoder revision |
| src/semcache/edgelora/r5_perf_runtime.py | Fresh block process/session lifecycle and isolated warmup |
| src/semcache/edgelora/r5_perf_analysis.py | Independent audit, fidelity, MISS controls and cycle-level statistics |
| scripts/112_run_r5_balanced_benchmark.py | Dry-run, diagnose, fixture, opt-in run and analyze |
| scripts/113_analyze_r5_balanced_benchmark.py | Independent read-only analysis entrypoint |
| jobs/r5_opt_balanced_benchmark.sh | Unsubmitted SERAPH diagnostic/performance modes |
| tests/test_r5_perf_evidence.py | Accounting, overflow, multi-writer, corruption and numerical compatibility |
| tests/test_r5_perf_schedule.py | Order, bootstrap and qualification tests |
| tests/test_r5_perf_processes.py | Actual isolated ABBA TCP generation and adversarial artifact analysis |
| docs/R5_I3P_PERFORMANCE_CORRECTION.md | Design, attribution, operations, results and limitations |

Two existing R5 files were modified additively: `r5_generation.py` adds optional
timing spans; `r5_analysis.py` adds an explicit performance-block audit option for
10 requests instead of its unchanged default20. No frozen R4 file, Transformer
forward, LoRA arithmetic, weights, matching policy or cache format was changed.
