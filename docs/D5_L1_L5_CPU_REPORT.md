# CPU completion report — PARTIAL_BLOCKED

Date: 2026-11-13. No GPU inference, Slurm submission/cancellation/query, external
network/API, package installation, model download, Git commit or push occurred.
Only local TCP loopback tests used execution outside the socket-restricted sandbox.

## Actual source and protection

Worktree `/home/songsong/GP_semcache_lightperf`, branch
`exp/d5-l1-l5-lightperf`, HEAD
`4916836ce328e119f3a8d85c6c1afc7cf8d363e5`. HEAD is exactly the requested
successful local Strict commit; no remote comparison was made. Initial and final
tracked source diffs are empty. All new Light implementation files are untracked
and uncommitted. A clean tracked status alone is not an approved Light source;
production pinning rejects these uncommitted new files.

Other registered worktree: `/home/songsong/SemCache`, Strict branch at `55587ba…`.
Its old local logs were read only. `/data/khuss/repos/GP_semcache` is absent here.
No original code/CLI/job/test/analyzer/auditor was changed: **0 modified Strict
lines**. No branch/worktree operation or Git source update was performed.
Old sealed GPU results and their manifests are not present on this host, so a
complete historical re-hash cannot be claimed. No writes target their locations.
New fresh outputs are restricted to resolved, non-symlink paths below
`results/r5/d5_lightperf`, with input overlap rejection.

## Implemented files and their limits

| New file | Implemented behavior |
|---|---|
| `src/semcache/edgelora/r5_d5_lightperf.py` | CPU prepare/reconstruct/inspect, explicit dependency/offline guards, development audit, hard-blocked run, receipt table exports |
| `r5_d5_lightperf_io.py` (same source directory) | Separate format/mode, incremental bounded summary sink, no tensor/blob API, fresh path checks, SHA256 manifests, new worktree/branch/commit/content pinning |
| `r5_d5_lightperf_workload.py` | Original first-occurrence NATURAL prefixes, whole groups, duplicate/routing/policy checks, explicit new continuous-session contract, conservative preflight, CPU-only 324-condition candidate matrix |
| `r5_d5_lightperf_emulation.py` | Actual framed-byte chunk pacing in both sender directions, preserved frozen Channel, bounded ledger retention, measured per-operation LoRA service emulation, separate UD sparse endpoint |
| `r5_d5_lightperf_timing.py` | Opt-in native generation kernel using original physical/protocol/impact primitives, no raw logits, bounded wall components, lifecycle ACK/STOP/BYE state contract |
| `r5_d5_lightperf_analysis.py` | Receipt/source/plan/audit/output/row/latency/resource verification, descriptive mean/median/nearest-rank P95, comparable identity checks, EXACT delta/speed ratio and separate APPROX labels, measured crossing bracket helper |
| `scripts/159_r5_d5_lightperf.py` | Separate CLI; Strict defaults unchanged |
| `jobs/r5_d5_lightperf.sh` | Manual template that deliberately exits BLOCKED; syntax checked only |
| `tests/test_r5_d5_lightperf.py` | 26 stdlib tests; real loopback and original PhysicalOwner fixtures; no optional test skips |
| `docs/D5_L1_L5_DESIGN.md` | Read-only audit, actual call graph and necessary computation vs observation vs safety classification |
| `docs/D5_L1_L5_GPU_RUNBOOK.md` | Exact defaults/schema, future manual three-dataset five-request PLAN/CPU commands, launch blockers and later gates |
| This report | CPU evidence and explicit incomplete acceptance criteria |

The native kernel imports torch lazily, so its real model path has **not** executed
here. This is more limited than an end-to-end Light runner. The physical fixture
compares the actual Strict dense Evidence sink against evidence=None plus Light
profiling, with identical original physical operator/proofs. It checks hashes,
EXACT/APPROX decisions, fresh/skipped rows, occupancy, insertion/eviction and PBR
under a small fixture capacity/explicit PBR trigger. The greedy-loop fixture checks
generated token equality and termination. These are not native OPT fidelity tests.

## L1–L5 readiness

| Layer | CPU implementation | Actual launch/evidence readiness |
|---|---|---|
| L1 | Light kernel, sink, wall profiler and verifier prototypes implemented | **BLOCKED:** guarded ES/UD process orchestration, compact full request/run receipts, timing integration and native Light parity missing |
| L2 | 3 datasets / 5,20,50 selection and continuity contracts; static budgets implemented | **BLOCKED:** real assets and actual sample IDs unavailable; real prefixes/Strict selection match NOT_VERIFIED; no 50-request runtime proof |
| L3 | Unlimited/200/100/50 framed-byte shaping, bilateral loopback tests | CPU fixture verified; production worker wiring and actual achieved link rates NOT_VERIFIED; RTT/shared-link scheduler NOT_IMPLEMENTED |
| L4 | 1/.5/.25 measured operation service emulation; same real scalar LoRA arithmetic and loopback endpoint | CPU fixture verified; native torch isolation/performance and ES/UD worker integration NOT_VERIFIED; hardware equivalence NOT_ESTABLISHED |
| L5 | Explicit CPU candidate matrix, comparison/statistics and crossing bracket helper, JSON/CSV/Markdown table pipeline | **PARTIAL:** authenticated measured-series grouping, calibration receipt binding and figure export NOT_IMPLEMENTED; no measured crossing |

No CPU approval suitable for GPU launch is issued by this revision, even if
dependencies become available. `cpu-validate` cannot silently turn Strict native
tests and Light scalar fixtures into native Light approval. `run`, `inspect` and
the job template do not present a READY state. Future native worker implementation
and native Light parity tests are required, not merely filling path variables.

## Tests actually executed

All test commands had `CUDA_VISIBLE_DEVICES=''` and `PYTHONPATH=src:tests`.
Elapsed times below are unittest's measured suite wall times.

Command A:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest test_r5_d5_lightperf -v
```

| Attempt / change | Run | PASS | FAIL | ERROR | SKIP | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| Initial sandbox attempt | 23 | 21 | 0 | 2 | 0 | 0.454 |
| Same command, local socket permission granted | 23 | 23 | 0 | 0 | 0 | 0.562 |
| Added actual Strict sink and fixture eviction/PBR parity | 23 | 23 | 0 | 0 | 0 | 0.854 |
| Added receipt tamper/old approval/prepared input tests | 26 | 26 | 0 | 0 | 0 | 1.584 |
| Final verifier integer/finite quality checks | **26** | **26** | **0** | **0** | **0** | **0.691** |

The initial two errors were `PermissionError` at socket creation. They were not
skipped or converted to PASS; the identical command was rerun with approved local
loopback execution. No external network was needed.

Command B (existing modules unchanged):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest \
  test_r5_natural_shutdown test_r5_natural_phase3a \
  test_r5_subsequence_generation test_r5_subsequence_validation \
  test_r5_sparse_physical test_r5_subsequence_audit \
  test_r5_natural_approx test_r5_subsequence_natural -v
```

**110 run, 102 PASS, 0 FAIL, 0 ERROR, 8 SKIP, 10.885 s.**
Skipped native cases: two installed OPT generation/impact tests, native validation
mode generation parity, torch alias/data mutation, actual bare Linear hook, native
attention reducer, native APPROX TCP smoke, native NATURAL TCP smoke. Dependencies
torch/transformers/numpy are absent; MultiNews rouge scorer is absent too. Thus the
required all-native zero-skip acceptance gate is **not met**. No installation was
attempted. Existing Strict source pinning still rejects this Light branch as before.

Additional checks: new Python files compile; shell template passes `bash -n`;
`git diff --exit-code` and `git diff --check` pass for existing tracked files.
CLI environment/matrix/development fingerprint were inspected, with no GPU action.

## Inspected dry-run plans and resources

Artifacts: `results/r5/d5_lightperf/cpu_review_20261113_01/`.
This sealed development directory contains environment discovery, 9 **synthetic**
stream lists for the three datasets × 5/20/50, the 12 knob pairs, and 324 per-
condition static resource fixture rows. Its six files total 541838 bytes;
manifest SHA256 is
`603bc3404dc6df6252a5f5d95788e11f241ee84b3916e3efa78d9b5d7015aefe`.
Real validated streams = **0**; selected/submitted jobs = **0**.

The synthetic stream tests use stable 5⊂20⊂50 prefixes and explicit cross-block
contracts. Their `S0…S49`/`P000000T…` IDs are not real dataset selections. The resource
fixtures deliberately use 512 input tokens and the existing 32/96 output caps;
they are not valid retokenized real task lists. Real preparation does not use
these artificial resource fixtures and never mutates original prompts.

Unlimited bandwidth / factor1, **unmeasured** 5s per forward plus 600s reserve:

| Dataset | N | Max forwards | Predicted condition seconds | Static time result |
|---|---:|---:|---:|---|
| qasper | 5 | 160 | 1400 | Fits static bounds, runtime checks pending |
| qasper | 20 | 640 | 3800 | Fits static bounds, continuous contract required |
| qasper | 50 | 1600 | 8600 | BLOCKED_INFEASIBLE |
| 2wikimqa | 5 | 160 | 1400 | Fits static bounds, runtime checks pending |
| 2wikimqa | 20 | 640 | 3800 | Fits static bounds, continuous contract required |
| 2wikimqa | 50 | 1600 | 8600 | BLOCKED_INFEASIBLE |
| multi_news | 5 | 480 | 3000 | Fits static bounds, runtime checks pending |
| multi_news | 20 | 1920 | 10200 | BLOCKED_INFEASIBLE |
| multi_news | 50 | 4800 | 24600 | BLOCKED_INFEASIBLE |

Across all fixture knobs/conditions: 60 rows pass static bounds with runtime checks
pending, 264 block. Slowed UD cases lack authenticated calibration; .5/.25 plans
with no operation bound block. Network forecasts include both directions, maximum
headers/framing and controls, without assuming cache HITs. Runtime estimates may
be longer due to proofs, semantic processing, quality and model loading.

Preflight also reports max context/forwards, per-adapter request counts/capacity,
host reserve, VRAM/attention working sets, ≥12 GiB free-after-load requirement,
≥32 GiB available host RAM, ≥4 CPUs, disk/file bounds, framing wire bounds and
fixed worker/Slurm deadlines. It never increases resources or reshapes workloads
to turn a failed 50-request plan into PASS. No physical GPU resources were measured.

## Source approval and next steps

Final development metadata and measured local CPU host resources are saved in
`results/r5/d5_lightperf/cpu_review_final_20261113_01/`. These receipts explicitly
do not authorize GPU work. See the runbook's three-dataset manual commands.
After remaining implementation, the user must review/authorize a commit, obtain
a fresh content fingerprint, prepare new real five-request plans, and run fresh
zero-skip native Light CPU approval. Any changed source/commit/assets require new
plans and audits. No old CPU approval is reusable as a new commit's approval.

Keep L2 20/50 and L3/L4/L5 on hold until separately approved five-request GPU
evidence exists. Historical user-supplied Strict reference is 221601 files /
34.866 GB; MultiNews EXACT 2/5 HIT, 2880 QKV rows, fidelity 5/5; APPROX 2/5 HIT,
10944 rows, fidelity 3/5. These values were not independently reproduced here.
Strict is independent value-audit observation; Light is a different low-persistence
measurement mode. They do not form an observer-equivalent speed comparison.

**Final conclusion: PARTIAL_BLOCKED.** CPU fixtures do not prove GPU speedup,
native correctness/safety, full-vocabulary numerical fidelity or successful
50-request continuity. `independent_full_vocabulary_audit=NOT_EVALUATED` and
`correctness_qualified_speedup=NOT_ESTABLISHED` remain mandatory.
