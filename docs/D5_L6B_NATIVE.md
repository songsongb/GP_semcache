# D5 L6-B on SERAPH

Use one interactive Bash shell and the semcache interpreter throughout preparation, submission and analysis. Start after the new source has been committed and reviewed. Historical v5 and L6-A approvals are reference evidence only.

```bash
cd /data/khuss/repos/GP_semcache_lightperf
PY=/data/khuss/anaconda3/envs/semcache/bin/python3
CLI=scripts/165_r5_d5_l6b.py
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
if [[ -z ${HF_HOME:-} ]]; then read -er -p 'Existing offline HF_HOME: ' HF_HOME; fi
if [[ -z ${TMPDIR:-} ]]; then read -er -p 'Existing writable TMPDIR: ' TMPDIR; fi
export HF_HOME TMPDIR
export HF_HUB_CACHE="$HF_HOME/hub"
git status --short --branch
"$PY" scripts/159_r5_d5_lightperf.py fingerprint
"$PY" "$CLI" --help
"$PY" "$CLI" check
```

Review the printed source and fingerprint. Enter the two independently reviewed values below. The native control rejects an uncommitted, dirty or changed executable source. Keep the trial ID printed by this command for all subsequent commands. If continuing in another shell, restore PY, CLI, the original offline paths and TRIAL from the original preparation.

```bash
read -r -p 'Reviewed commit: ' REVIEWED
read -r -p 'Reviewed content fingerprint: ' FP
TRIAL="long50_$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "$TRIAL"
"$PY" "$CLI" prepare-cpu --trial "$TRIAL" --reviewed-commit "$REVIEWED" --reviewed-fingerprint "$FP"
"$PY" "$CLI" inspect --trial "$TRIAL"
```

Preparation creates `results/r5/d5_l6a/` directories named with the trial ID plus `_plan` and `_cpu`. It reconstructs the plan from the frozen v5 cell. It then creates `results/r5/d5_lightperf/l6b/$TRIAL/{plan,cpu}`, runs the native approval suite (including both L6 modules) and records the two native inventories, Python, offline paths and native ID in ready.json. Native approval, parent inventory verification and the bound all-fresh forecast are checked again at launch. The isolated UD checks the sealed inventories without loading the model or tokenizer. The source fingerprint includes the new runner, tests and documentation.

Only after successful inspection and explicit authorization, the user runs the following **separate** command. This is the only public command that submits jobs.

```bash
"$PY" "$CLI" submit --trial "$TRIAL" --node moana-y3 --enable-approximate-experiment
```

The submission runs Slurm test-only and held submission for each condition, records the job IDs, and releases after registering all three. FULL, EXACT and APPROX run in order via afterok, one fresh 50-request process per job. A submission failure can leave held jobs; inspect job IDs and Slurm logs and cancel outstanding jobs manually. A consumed trial ID cannot be submitted again. Failure, time limit or a different GPU/affinity prevents continuation. Slurm cannot guarantee that the next job receives the same GPU UUID. The run records the actual allocation under the trial and requires the same node, UUID and four CPU indices. The actual Slurm EndTime is checked before the probe, before the condition, and inside the parent and ES after approval. The batch wrapper locates the repository via the submitted working directory.

```bash
"$PY" "$CLI" status --trial "$TRIAL"
"$PY" "$CLI" reconcile --trial "$TRIAL"
"$PY" "$CLI" analyze --trial "$TRIAL"
```

Status reads scheduler accounting. Reconcile additionally writes fresh terminal-failure records for scheduler failures including cancellation and timeout, which can kill the worker before its exception handler runs. A missing accounting record is reported as pending. Analyze requires all three registrations, condition completion records, three separate job IDs, 50 verified rows per run, matched backend/GPU/concurrency/workload and EXACT 50/50 token agreement. It publishes the L6-A JSON, two 15-row CSVs and sealed inventory in the L6-A namespace with suffix `_analysis`, then records `matched.json` in the trial. Treat incomplete directories as failed unpublished output. Preserve every attempt.

The 50 Mbps value controls application loopback pacing and the .25 factor controls sleep-based UD service emulation. The 10 original requests repeat five times with no cycle reset. Each adapter has at most 64 resident entries and 512 MiB; the 128-entry CPU configuration is excluded. Shared inference and Impact algorithms use the existing native implementation. Results are single-trial descriptive and correctness-qualified speedup is NOT_ESTABLISHED. Report scorer means and APPROX token agreement independently with the eight-token cap.

### Local CPU regression command

From the repository root, the following selects the native L6-B approval modules and the v5 sweep regression module. The localhost TCP fixtures need loopback socket permission. A local unittest result does not produce a native CPU approval record.

```bash
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:tests
.venv-d5l2e/bin/python -c 'import torch,sys,unittest; from semcache.edgelora.r5_d5_lightperf import approval_modules; from semcache.edgelora.r5_d5_lightperf_runtime import native_versions; assert not torch.cuda.is_initialized(); print(native_versions(),flush=True); names=approval_modules(dict(workload_mode="CONTINUOUS_REPLAY",enable_continuous_replay=True,impact_implementation="MASK_INDEX_REUSE_PER_LAYER_V1",l6_binding=True))+["test_r5_d5_sweep"]; r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames(names)); assert not torch.cuda.is_initialized(); sys.exit(not r.wasSuccessful() or bool(r.skipped))'
```

The final focused run used the same environment and these arguments:

```bash
.venv-d5l2e/bin/python -c 'import torch,sys,unittest; assert not torch.cuda.is_initialized(); r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])); assert not torch.cuda.is_initialized(); sys.exit(not r.wasSuccessful() or bool(r.skipped))' test_r5_d5_l6b test_r5_d5_l6a test_r5_d5_sweep
```

Local development evidence on base HEAD 044f2254: the broader run before the last four runner tests passed 280 tests in 518.976 seconds. The final focused run passed 56 tests in 6.141 seconds (8 L6-B, 12 L6-A, 36 sweep). Both had zero failures, errors and skips. The local checkout had no SERAPH reference inventory; `check` exited 2 with BLOCKED_MISSING_REFERENCE. No real PLAN, CPU approval, scheduler registration or GPU measurement was produced.
