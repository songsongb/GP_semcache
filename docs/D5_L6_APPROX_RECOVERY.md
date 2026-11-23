# L6-B APPROX-only recovery

This path reuses sealed FULL job 223161 and EXACT job 223163 under long50_20261123T061210Z at commit 513b9610488c11ee8aa9a538f7829ec5d1490360. The historical GPU is 0dff7717-a97a-b957-b478-f26213d1b38e and historical affinity is [40,41,88,89]. Both runs must verify with their original native PLAN, CPU audit, complete records and job registrations. EXACT must agree with FULL on all 50 token sequences. The historical failed APPROX stays in its original directory.

Recovery publishes a new native PLAN, CPU audit and one APPROX job in a fresh trial. It runs the existing L6-A and native CPU approvals for the new clean reviewed source. The recovery contract binds historical receipt inventories and registration records into the new plan digest. Native validation rechecks them. The historical approval is used only to verify its own runs.

The dependency check compares original and current file maps. Existing inference source and executable scripts/jobs must retain identical bytes except L6-B binding/runner control files and Light approval_modules. An AST comparison enforces that Light edits are confined to approval_modules. New recovery code and tests are included in the new fingerprint. Review these allowed control changes as part of source review.

The recovery runs the original 50 requests, five cycles, eight-token cap, 50 Mbps application loopback pacing, .25 sleep-based UD service factor, 64 entries per adapter and 9000-second deadline. It starts fresh cache owners once and preserves them throughout all 50 requests. Four currently allocated CPU indices are selected. Topology records include package/core IDs, sibling sets, NUMA membership, CPU model/features, kernel, affinity and execution environment. The node and measured GPU UUID must match the historical run before inference.

## SERAPH preparation

After committing and reviewing the new source, use the same semcache interpreter throughout.

```bash
cd /data/khuss/repos/GP_semcache_lightperf
PY=/data/khuss/anaconda3/envs/semcache/bin/python3
CLI=scripts/166_r5_d5_l6_recovery.py
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
read -er -p 'Existing offline HF_HOME: ' HF_HOME
read -er -p 'Existing writable TMPDIR: ' TMPDIR
export HF_HOME TMPDIR
export HF_HUB_CACHE="$HF_HOME/hub"
git status --short --branch
"$PY" scripts/159_r5_d5_lightperf.py fingerprint
"$PY" "$CLI" check
read -r -p 'Reviewed commit: ' REVIEWED
read -r -p 'Reviewed fingerprint: ' FP
TRIAL="approx_recovery_$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "$TRIAL"
"$PY" "$CLI" prepare-cpu --trial "$TRIAL" --reviewed-commit "$REVIEWED" --reviewed-fingerprint "$FP"
"$PY" "$CLI" inspect --trial "$TRIAL"
"$PY" "$CLI" dry-run --trial "$TRIAL"
```

CPU dry-run prints submission and test-only argv without invoking Slurm. New trial records are under results/r5/d5_lightperf/l6b followed by the trial ID. New L6-A inputs use that ID with _plan and _cpu suffixes under results/r5/d5_l6a. The recovery CLI invokes no FULL or EXACT GPU work.

## Explicit submission and analysis

After separate user authorization, submit APPROX only.

```bash
"$PY" "$CLI" submit --trial "$TRIAL" --enable-approximate-experiment
"$PY" "$CLI" status --trial "$TRIAL"
"$PY" "$CLI" reconcile --trial "$TRIAL"
"$PY" "$CLI" analyze --trial "$TRIAL"
```

Submit runs real Slurm test-only, held submission, registration and release. It targets the historical node. Another GPU causes rejection. Submission consumes the trial ID. There is no automatic retry. A failed submission can leave a held job for explicit cancellation. Reconcile records scheduler cancellation/timeout in the recovery directory. Preserve failed attempts.

Analysis verifies each run with its own sealed plan and CPU audit. The new APPROX must complete 50/50 and pass row/QKV accounting and inventory verification. Cycle/cumulative outputs each contain 15 rows. They report token agreement, available timing, quality, wire bytes, QKV ratios, occupancy and request-level policy deltas. The output lives in results/r5/d5_l6a under the recovery ID with suffix _recovery_analysis and contains long_replay_analysis.json, cycle_metrics.csv, cumulative_metrics.csv and light_inventory.json.

Status is LIMITED_COMPARABILITY and scope CROSS_TRIAL_RECOVERY_DESCRIPTIVE. The report preserves strict compare() output without substituting source or CPU fields. It writes no MATCHED_COMPLETE record. Source control code, cores and allocation differ, and historical NUMA snapshots were not captured. Observed condition-minus-FULL E2E differences are exported, while speedup stays null. Correctness-qualified speedup and quality preservation stay NOT_ESTABLISHED. Eight-token APPROX agreement does not establish dataset quality preservation.

## Local development verification

Base branch is exp/d5-l1-l5-lightperf, HEAD 513b9610488c11ee8aa9a538f7829ec5d1490360. The original trial is absent from this local checkout. The recovery check reports BLOCKED_MISSING_REFERENCE and exits 2. No real recovery plan, CPU approval or scheduler job was created.

Local dependencies were Python 3.10.12, Torch 2.14.1+cpu, Transformers 4.57.6 and NumPy 2.2.6. The native CPU approval modules plus sweep passed 296 tests in 433.532 seconds under the initial test layout, which included six imported fixture tests also loaded elsewhere. After removing that duplicate discovery and completing the recovery worker test, the focused run passed 63 tests in 8.963 seconds (7 recovery, 8 L6-B, 12 L6-A and 36 sweep), with zero failures, errors or skips. A final recovery-only run after the package-version check passed 7 tests in 0.089 seconds. Counts across runs overlap.

The focused command was:

```bash
env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:tests .venv-d5l2e/bin/python -c 'import torch,sys,unittest; assert not torch.cuda.is_initialized(); r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])); assert not torch.cuda.is_initialized(); sys.exit(not r.wasSuccessful() or bool(r.skipped))' test_r5_d5_l6_recovery test_r5_d5_l6b test_r5_d5_l6a test_r5_d5_sweep
```

Both the native run and focused command assert CUDA is uninitialized before and after testing. Native fixtures use localhost TCP and need loopback permission. Recovery scheduler calls, submission/release, UUID checks and GPU probing are mocked. The shell wrapper passed bash -n and source changes passed git diff --check. These are DEVELOPMENT_TESTS_PASS results; clean SERAPH CPU approval remains required.
