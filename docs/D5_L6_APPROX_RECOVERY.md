# L6-B APPROX-only recovery

This path reuses sealed FULL job 223161 and EXACT job 223163 under long50_20261123T061210Z at commit 513b9610488c11ee8aa9a538f7829ec5d1490360. The historical GPU is 0dff7717-a97a-b957-b478-f26213d1b38e and historical affinity is [40,41,88,89]. Both runs must verify with their original native PLAN, CPU audit, complete records and job registrations. EXACT must agree with FULL on all 50 token sequences. The historical failed APPROX stays in its original directory.

Recovery publishes a new native PLAN, CPU audit and one APPROX job in a fresh trial. It runs the existing L6-A and native CPU approvals for the new clean reviewed source. The recovery contract binds historical receipt inventories and registration records into the new plan digest. Native validation rechecks them. The historical approval is used only to verify its own runs.

The dependency check compares original and current file maps. Existing inference source and executable scripts/jobs must retain identical bytes except L6-B binding/runner control files and Light approval_modules. An AST comparison enforces that Light edits are confined to approval_modules. New recovery code and tests are included in the new fingerprint. Review these allowed control changes as part of source review.

The recovery runs the original 50 requests, five cycles, eight-token cap, 50 Mbps application loopback pacing, .25 sleep-based UD service factor, 64 entries per adapter and 9000-second deadline. It starts fresh cache owners once and preserves them throughout all 50 requests. Four currently allocated CPU indices are selected. Topology records include package/core IDs, sibling sets, NUMA membership, CPU model/features, kernel, affinity and execution environment. The node must match the historical run before inference. STRICT_UUID remains the default policy. SAME_RTX3090_SPEC requires a separately approved PLAN and explicit --allow-compatible-gpu at preparation and submission.

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

Submit runs real Slurm test-only, held submission, registration and release. It targets the historical node. Under the default policy another UUID causes rejection. Submission consumes the trial ID. There is no automatic retry. A failed submission can leave a held job for explicit cancellation. Reconcile records scheduler cancellation/timeout in the recovery directory. Preserve failed attempts.

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


## GPU allocation diagnostic and explicitly approved replacement

The allocation request uses --nodelist and --gres=gpu:1. These constrain the node and GPU count, with no requested physical UUID. The cluster configuration was not available in the local checkout. The runtime probe must therefore determine the allocated UUID. According to the [SchedMD GRES documentation](https://slurm.schedmd.com/gres.html), GPU requests specify resource type/count and --gpu-bind binds tasks to GPUs. Binding is not a mechanism to acquire an unallocated device. The runner adds no --gpu-bind or GPU-index workaround.

Before inference, every recovery probe publishes SEMCACHE_D5_APPROX.gpu_diagnostic.json in the new trial. It records expected and actual UUID, name, total/free memory, node, CPU count, affinity, CUDA device count, CUDA_VISIBLE_DEVICES, SLURM_JOB_GPUS, SLURM_STEP_GPUS, software versions, selected policy, stderr and failure reasons. Probe failure, malformed output and timeout produce a BLOCKED diagnostic. The file is write-once. Historical receipts and failed trial records are never updated.

If a different physical GPU is explicitly authorized, choose a fresh trial ID and create its PLAN and approvals using this variant of the preparation commands above:

```bash
TRIAL="approx_replacement_$(date -u +%Y%m%dT%H%M%SZ)"
"$PY" "$CLI" prepare-cpu --trial "$TRIAL" --reviewed-commit "$REVIEWED" --reviewed-fingerprint "$FP" --allow-compatible-gpu
"$PY" "$CLI" inspect --trial "$TRIAL"
"$PY" "$CLI" dry-run --trial "$TRIAL" --allow-compatible-gpu
```

Following separate submission authorization:

```bash
"$PY" "$CLI" submit --trial "$TRIAL" --enable-approximate-experiment --allow-compatible-gpu
"$PY" "$CLI" analyze --trial "$TRIAL"
```

The PLAN digest binds SAME_RTX3090_SPEC, and submit/dry-run must confirm the same opt-in. The worker requires one CUDA device, the original node, NVIDIA RTX 3090 (excluding Ti), exactly the original total VRAM, at least 12 GiB free GPU memory, the original native package versions and compiled CUDA version, and the original cuDNN version when recorded. The normal model-load reserve and 22 GiB peak limits still apply. The actual replacement UUID is passed to native execution and recorded without rewriting it. No historical driver-version assertion is possible because the original driver version was not measured.

Cross-trial output remains LIMITED_COMPARABILITY with strict compare() unchanged, speedup null and correctness_qualified_speedup NOT_ESTABLISHED. The report adds physical_gpu_uuid_changed. The old failed recovery job 223325 and its trial must be retained. A new source commit/fingerprint and fresh L6-A/native CPU approvals are required before using either policy in a new trial.

### GPU policy revision CPU verification

The revision starts at HEAD 3ba8d108de2df274d23bf53ccc35af8f5a6e8fc6 on exp/d5-l1-l5-lightperf. Light/Worker/Replay plus L6 recovery/L6-B/L6-A/v5 sweep passed 145 tests in 9.320 seconds. The final focused configuration, including additional cuDNN/probe-error and replacement-UUID analysis cases, passed 68 tests in 6.511 seconds. All runs had zero failures, errors and skips. Counts overlap. CUDA was hidden and asserted uninitialized before/after both runs. The extended regression used the same command shown above with test_r5_d5_lightperf, test_r5_d5_lightperf_worker and test_r5_d5_lightperf_replay added to the module arguments.

The final focused tests include 12 recovery tests, 8 L6-B tests, 12 L6-A tests and 36 v5 sweep tests. They exercise the CPU dry-run with compatible-GPU opt-in and assert zero scheduler invocations, strict UUID rejection, opt-in replacement acceptance, model/VRAM/device-count/free-memory rejection, diagnostic persistence before inference, probe timeout/malformed output, software mismatch, immutable original fixtures and LIMITED_COMPARABILITY for an actual replacement UUID in the analysis fixture. The CLI help, bash -n and git diff --check passed. No real Slurm test-only or GPU probe was executed locally. The original trial is absent, so CLI check returned BLOCKED_MISSING_REFERENCE (exit 2). New clean source approval and the SERAPH reference checks are still pending.
