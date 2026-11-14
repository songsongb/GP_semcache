# L2 continuous replay CPU report — PARTIAL_BLOCKED

## 1. Actual source and environment

Worktree `/home/songsong/GP_semcache_lightperf`, branch
`exp/d5-l1-l5-lightperf`, HEAD `12cb176dbbe411a6e7705c9a5f7cda4b396fd730` matches
the requested commit. Initial tracked tree was clean; `?? results/r5/` was already
present and was left alone. Follow-up changes are uncommitted. No commit/push,
branch/worktree modification, remote call, install/download, GPU inference or
Slurm query/submission/cancellation was performed.

The execution environment differs from SERAPH: `/data/khuss` is absent, native
torch/transformers/numpy/rouge are absent, and real I5/multi/base prepared manifests
found in this worktree =0. Real dataset IDs/pool sizes/approvals/GPU receipts are
not accessible. No new experiment/approval/output artifact was created; checks
used memory and stdout with bytecode writing disabled. Temporary-output suites
were not run outside the user-required `/data/khuss` namespace.

## 2. Changed/added files and reason

Added:

* `r5_d5_lightperf_replay.py`: explicit replay semantics, provenance, deterministic
  cycling, statistics, independent metadata/hash validation.
* `r5_d5_lightperf_replay_analysis.py`: descriptive session and five-request-window
  metrics, adapter/user reuse, HIT/MISS completion, component scope summaries,
  matched speed ratios/token agreement and no extrapolated break-even threshold.
* `tests/test_r5_d5_lightperf_replay.py`:20 memory-only replay/control/analysis tests.
* `jobs/r5_d5_lightperf_replay.sh`: one manual dataset/20-or-50/condition job;
  no array/sweep, replay+extended+GPU opt-ins, baseline knobs, fresh DATA paths.
* `docs/D5_L2_REPLAY_RUNBOOK.md` and this report.

Updated Light-only modules under `src/semcache/edgelora`:

* `r5_d5_lightperf_workload.py`: extracts the original pool validator without
  changing NATURAL duplicate/group/routing policy or resource formulas.
* `r5_d5_lightperf.py`: replay/baseline options, reconstruction and required test
  coverage, stdout CPU preflight, position analysis exports, plan JSON bound.
* `r5_d5_lightperf_runtime.py`: extra replay launch opt-in and per-occurrence
  receipt binding; existing serial/native execution and resource guards retained.
* `r5_d5_lightperf_receipts.py`: bounded occurrence provenance in compact receipts.
* `r5_d5_lightperf_analysis.py`: explicit replay receipt validation, mode isolation,
  compatible legacy NATURAL receipt checks and matching replay contract hashes.
* `tests/test_r5_d5_lightperf_native.py`: real Tiny OPT20/50 repeated execution
  through the shared serial engine, Strict/Light RAM logit and cache comparisons.

The existing script159 and baseline five-request job are reused unchanged. No
Strict source/CLI/job/test/formula/configuration or sealed evidence was edited.
Diff paths intersected with the Strict baseline4916836 tree = **empty**.

## 3. Workload/continuous session contract

NATURAL remains the default and forbids duplicate original samples. Replay needs
an explicit mode and prepare opt-in; runtime additionally needs explicit replay
and extended approvals. The source pool is all validated dataset repetition-0
NATURAL tasks in original order. No workload ranking uses HIT, lengths, user,
adapter or measured cost. Each pool is consumed before cycling; no source request
is duplicated within that original pool.

First-cycle tasks, including execution IDs, are untouched. Repeated occurrences
change only execution request_id; original sample, request sequence field, user,
adapter, prompt, tokens, prepared sample, generation and seed remain. Parallel
provenance records original ID/block/index/task SHA256, cycle/occurrence index
and global session ordinal. All execution IDs are unique, bounded protocol-safe
strings. Partial-cycle/group boundary errors fail without reordering/truncating.

The cache lifetime is N requests within one condition, with independent original
adapter namespaces and persistent ES owners/indices/UDs. The adapter's last ACK
permits STOP/BYE before long local admission, preserving the earlier shutdown
fix. Conditions start empty. Original cold blocks are explicitly labeled as
sources of a **new** CONTINUOUS_REPLAY session, not an original continuous trace.

`--baseline-plan` compares the frozen Light first five tasks, seed and input/
model-adapter identity; its inventory is pinned. Missing/mismatched baseline is
NOT_MATCHED with reasons, never an automatic parity claim. Strict pilot matching
remains a separate optional historical selection check.

## 4. Three-dataset5/20/50 construction checks

Memory fixtures for all three dataset names use an explicitly synthetic ten-task
original pool with original deterministic routing. Results (not real selections):

| Each dataset fixture | N | Unique samples used | Repeated occurrences | Repeat ratio | Full cycles |
|---|---:|---:|---:|---:|---:|
| qasper /2wikimqa /multi_news |5|5|0|0%|0, then5 tasks of first cycle|
| qasper /2wikimqa /multi_news |20|10|10|50%|2|
| qasper /2wikimqa /multi_news |50|10|40|80%|5|

5⊂20⊂50 prefixes, full source task restoration, routing, unique execution IDs,
occurrence ordinals and source hashes pass. A three-source partial-cycle fixture
also passes with an explicit mismatch label; group-splitting fixtures reject.
Actual dataset source pool sizes are **unknown here**, not assumed to be10.

## 5. CPU tests and limitations

All executed test commands used
`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests`.

* Initial replay module:15 PASS/0 FAIL/0 ERROR/0 SKIP,0.387s.
* Replay module after source/receipt checks:17 PASS/0 FAIL/0 ERROR/0 SKIP,0.933s.
* Replay+selected existing regression:54 PASS/0 FAIL/0 ERROR/0 SKIP,0.687s.
* Replay+selected regression before output-protection check:56 PASS/0 FAIL/0 ERROR/0 SKIP,0.820s.
* Final replay+selected regression: **57 PASS/0 FAIL/0 ERROR/0 SKIP,0.813s**.
* Native module final run: **0 PASS/6 FAIL/0 ERROR/0 SKIP,0.005s**. Every case fails at
  the explicit missing-package gate; no native model forward ran. Initial native
  invocations had the same six dependency-gate failures in0.004s and0.003s.

Final regression selection:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest \
 test_r5_d5_lightperf_replay test_r5_d5_lightperf.WorkloadTests test_r5_d5_lightperf.TimingTests \
 test_r5_d5_lightperf_worker.GPUUUIDValidation test_r5_i5_gpu \
 test_r5_d5_lightperf_worker.WorkerContracts.test_allocation_requires_every_guard_without_gpu_import \
 test_r5_d5_lightperf_worker.WorkerContracts.test_approx_extended_count_and_gpu_uuid_gates \
 test_r5_d5_lightperf_worker.WorkerContracts.test_job_template_missing_approval_exits_before_python_gpu \
 test_r5_d5_lightperf_worker.WorkerContracts.test_finalization_never_masks_primary_failure \
 test_r5_d5_lightperf_worker.WorkerContracts.test_reap_escalates_only_its_own_child \
 test_r5_d5_lightperf_worker.WorkerContracts.test_stop_requires_ack_and_duplicate_bye_fails \
 test_r5_d5_lightperf_worker.WorkerContracts.test_invalid_bye_cannot_report_success \
 test_r5_d5_lightperf_worker.WorkerContracts.test_logit_observer_is_bounded_memory_hook_without_result_blob -v
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest test_r5_d5_lightperf_native -v
```

Scalar ArrayTensor tests use the original PhysicalOwner, scoped proofs and CPU
LoRA arithmetic, verify repeated reuse, capacity, adapter ownership and empty
condition reset. Their attention values are explicitly stubbed; this is **not**
native attention/fidelity/GPU proof. Memory byte-store tests exercise new receipt
field/provenance binding and old five-request compatibility; they do not replace
filesystem seal tests. Full filesystem/loopback/native suites remain for SERAPH
CPU approval using TMPDIR under DATA. No SKIP was manufactured as PASS.

Native replay tests are implemented for20/50 across FULL/EXACT/APPROX with real
Tiny OPT and spawned CPU UDs, RAM logit parity, unique execution IDs, ten repeated
samples, row/route/provenance checks and final ACK/STOP/BYE. They remain **NOT_RUN
beyond dependency gate** locally. Source review is still mandatory. AST parsing,
both job syntax checks and git diff whitespace checks pass.

## 6. Existing five-request regression

The selected existing workload, greedy generation, latency-scope, lifecycle,
UUID/device/class/fingerprint and opt-in contracts pass. NATURAL returns the
same task bodies, rejects duplicates and does not cycle. Historical NATURAL
approvals keep their module list for offline verification; replay approvals add
the replay module. Runtime source pinning still requires fresh plans/approvals
for this new commit. The user's successful5-request GPU results/134-test CPU
approval were **not re-executed or reissued locally**.

## 7. Actual20/50 CPU preflight

| Actual dataset |20 requests|50 requests|Exact reason|
|---|---|---|---|
|qasper|BLOCKED_ENVIRONMENT|BLOCKED_ENVIRONMENT|No native packages, real I5/dataset/multi assets or supplied local paths|
|2wikimqa|BLOCKED_ENVIRONMENT|BLOCKED_ENVIRONMENT|Same; frozen successful5-request plan/receipts absent locally|
|multi_news|BLOCKED_ENVIRONMENT|BLOCKED_ENVIRONMENT|Same, including rouge scorer absence|

Actual selected counts, IDs, user/adapter distributions, baseline agreement and
measured GPU feasibility are NOT_EVALUATED. No real PLAN or CPU approval was
generated. The new stdout `preflight` command will report these from authenticated
actual plans on SERAPH; its PASS is static only and never GPU approval.

## 8. Static resource fixtures and blocked conditions

Additional **synthetic resource fixtures only** use512 input tokens, original
32-output caps for Qasper/2Wiki and96 for MultiNews. No production task is changed.
All rows include every request/forward/exchange and assume no cache savings.

| Dataset fixture |N|Forecast seconds|Max exchanges|Wire bound bytes, both directions|GPU estimate GiB|Static outcome|
|---|---:|---:|---:|---:|---:|---|
|qasper|20|3800|20480|15146321712|11.104|Bounds pass, runtime checks pending|
|qasper|50|8600|51200|37864624272|11.104|BLOCKED_INFEASIBLE: DEADLINE_OR_SLURM_BOUND|
|2wikimqa|20|3800|20480|15146321712|11.104|Bounds pass, runtime checks pending|
|2wikimqa|50|8600|51200|37864624272|11.104|BLOCKED_INFEASIBLE: DEADLINE_OR_SLURM_BOUND|
|multi_news|20|10200|61440|21984675632|11.123|BLOCKED_INFEASIBLE: DEADLINE_OR_SLURM_BOUND|
|multi_news|50|24600|153600|54960509072|11.123|BLOCKED_INFEASIBLE: DEADLINE_OR_SLURM_BOUND|

Forecast =600s reserve +unmeasured5s/max forward. Host available32GiB,4 CPUs,
allocated64GiB, per-adapter cache512MiB, peak allocation<22GiB/free-after-load12GiB,
existing context/generation/IO bounds, worker7200s/Slurm16200s remain. Per-tuple
reports include user/adapter counts, receipt/file bounds and uncertainty. GPU
memory and walltime were not actually measured. These fixture blocks are **not**
claims about inaccessible actual datasets, whose frozen budgets may differ.
There is no limit inflation, generation shortening, request deletion or retry.

## 9–11. New PLAN, source/approval and manual GPU execution

See `D5_L2_REPLAY_RUNBOOK.md` for actual commands. Review/commit all new source
files first, then obtain a new clean fingerprint on SERAPH. Prepare each tuple
with explicit replay mode/opt-in and frozen baseline reference, run preflight,
zero-skip native CPU validation and inspect. Preserve all prior plans/audits/runs.
The extended manual script requires replay+extended+GPU approvals, expected GPU
UUID, node/allocation, source and audit fingerprints, DATA-rooted paths and
baseline knobs. One job is one dataset×scale×condition; no arrays or automatic
submissions. Verify FULL before choosing EXACT, then APPROX with its separate
opt-in. Any blocked resource result prevents launch.

## 12. Bottleneck analysis

`analyze --registry ... --output-dir FRESH_DATA_PATH` verifies all plan/CPU/run/UD
receipts before analysis. Existing outputs remain; new position_analysis.json
and position_windows.csv describe five-request bins, HIT/MISS completion,
input/output distribution, user/adapter mix, physical rows/ratios, policy and
inclusive/observed-exclusive components. Missing metrics stay NOT_MEASURED/null.
Matched speedup/token agreement is descriptive. No sign/threshold is fabricated
or extrapolated across heterogeneous windows; quality is never pooled by dataset.

## 13. Not verified / conclusion

**PARTIAL_BLOCKED**: replay construction and memory CPU contracts are implemented;
real dataset preflight, native replay parity, new clean committed source/CPU
approval and all GPU execution remain outstanding. SERAPH's first full execution
may reveal native/asset issues not exercised here. No GPU readiness is asserted.
L3/L4 interfaces remain; L2 script fixes unlimited/factor1. No sensitivity runs or
Cartesian campaigns were performed. Performance emulation is not hardware proof.

User-supplied, single-trial2Wiki5 baseline retained as reference (not re-audited):
FULL TTFT/E2E3407.97/4760.64ms; EXACT8220.16/14933.81ms,3/5 HIT,7488 skipped rows,
tokens5/5; APPROX8325.82/14981.52ms,3/5 HIT,9504 rows,tokens2/5. Supplied impact/
owner-completion totals and HIT/MISS costs motivate analysis; they were not new
measurements here. Independent numerical audit stays NOT_EVALUATED and
correctness_qualified_speedup stays NOT_ESTABLISHED.
