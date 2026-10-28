# R2-C2: Pilot-guarded profile capacity screen

The reported 18.788 GiB requirement versus 18.400 GiB headroom was a
**conservative preflight rejection**, not an actual CUDA OOM. The reported
successful L1024 Pilot peak (~8.73 GiB) and completed L512 traces are historical
user evidence, not measurements collected by this change.

Inspected checkout: `exp/raw-semcache-r2`, HEAD
`c1efbcef067ad89072a9e2021e23e631535fb39c`, initially clean. The actual R2-C2
SERAPH launcher was not present in this checkout. Existing launch files were
inspected; the new `jobs/r2c2_profile_remaining.sbatch` is specifically authorized
by the R2-C2 request. No old launcher or result is changed.

## Scope and defaults

The 90 runner adds `--profile-memory-policy conservative|pilot_guarded`.
**Default is conservative.** The old `memory_estimate()` formula, its headroom
comparison, source capacity check and inherited R1 source preflight are unchanged.
The guarded screen is opt-in and accepted **only for `--phase profile` on CUDA**.
It does not change Full/No-Reuse/RAW Forward, selection, assembly, attention
outputs, dtype, output logits, profiler ranges, timing boundaries or tolerance.
Latency phases do not install the new Pilot memory observer.

`--profile-memory-allowance-gib` defaults to 1 and
`--profile-memory-safety-gib` to 2. Margins must be finite and positive and are
profile-only options. They are provisional diagnostic budgets, **not guaranteed
profiler peak bounds**. Raising or lowering a margin is an explicit new setting.

## Memory calculation

Let B be measured Pilot allocated bytes immediately after releasing any previous
source (model/LoRA/semantic encoder remain resident), and M the maximum allocated
bytes observed over source preparation, structural gates and Pilot requests.

```text
pilot_increment = M - B
guarded_additional_required = pilot_increment + profiler_allowance + safety_reserve
current_available = driver_free + allocator_reserved - allocator_allocated
approve iff guarded_additional_required <= current_available
```

This charges the **entire** observed allocation increase even if a same-length
source is already resident for another coverage. There is no assumed subtraction
for resident source or reserved allocator blocks beyond the existing headroom
formula. B and M are allocated memory; they are not reserved memory, `nvidia-smi`
usage or all-GPU memory use. Other processes reduce current driver free memory.

Frozen target requests reset Torch peak counters. A **Pilot-only** observer saves
the peak immediately before every reset of the selected device, then delegates
to the original reset. It also samples at observer entry/exit. It adds no CUDA
synchronization and no profiling hooks; no mathematical executor code is patched.
The source-free baseline and cumulative maximum are carried across coverages
sharing that length/source. Thus each approval conservatively covers source build
and its completed same-length cases; it is not merely the final request's peak.
All patches are removed on success or failure.

Each CUDA Pilot condition records `pilot_memory` with schema
`r2c2-peak-across-resets-v1`, baseline, peak, increase, physical total memory,
sample count and measurement scope. CPU pilots cannot supply guarded evidence.

## Approval and evidence integrity

Guarded profile requires a completed Pilot **manifest and capacity artifact**:

- `experiment=R2-C1`, Pilot phase, both execution statuses `COMPLETE`, GPU
  execution recorded, and final model weights verified.
- Matching actual physical GPU UUID (with optional `GPU-` spelling normalized).
- Identical environment identity, full code SHA, implementation file hashes,
  model/adapter weight digest, model/dtype, adapter/backend/output settings,
  cache residency/capacity and seed. Revision/tokenizer/encoder/version identities
  remain part of the existing environment digest.
- Same requested L and coverage, `CAPACITY_VALIDATED`, `structural_valid=true`.
  Manifest and capacity condition records must agree.
- Measured new-schema memory evidence, positive baseline and increase,
  peak > baseline, peak <= device total, consistent increase and >=2 observations.
  Missing or invalid values are rejected, never replaced with zero.

The original post-build source/token/namespace/weight/coverage
`condition_identity` comparison still runs before target validation/profiling.
Catalog QKV physical capacity checks, inherited R1 preflight, canonical backing
guards and all QKV/Logit/row checks remain mandatory. The new policy replaces
only the outer conservative screen for its selected profile cases. An inherited
preflight may still reject; it is never silently overridden.

All relevant Python files and the new helper are in code/implementation identity.
**This change invalidates historical Pilot compatibility. Run a fresh Pilot on
the final code. Do not edit old JSON to force approval.** A Pilot containing any
failed condition is rejected by guarded mode, even if another condition passed;
use a fresh explicit valid subset when necessary.

FP16 Strict FAIL is allowed as a diagnostic outcome; it is not a capacity failure
or correctness-qualified speedup. `ORACLE_DIAGNOSTIC` and
`safe_reuse_claimed=false` remain unchanged; atol=rtol=1e-3.

## Artifacts and failures

The existing `preflight` field preserves the original conservative estimate.
`memory_preflight_audit` additionally records chosen policy, complete conservative
and guarded budgets, Pilot baseline/peak/increase, margins, current free/allocated/
reserved/total, available headroom, approval/rejection reason, input Pilot paths
and SHA256 hashes, weight/condition identity and evidence errors.

Real Torch CUDA OOM is `OOM`; a screen rejection is `MEMORY_PREFLIGHT_FAILED`.
Trace budget/export errors containing a trace failure code are `TRACE_FAILED`;
other trace/runtime exceptions retain their actual type/traceback as invalid
execution. Failure records include memory at failure and previous preflight audit.
There is no smaller-length/dtype/cache/attention retry. Completed artifacts survive.

`91_validate_r2c2_results.py` returns nonzero for partial/failed runs, missing
profiles, unavailable profiler or changed code/GPU. It prints **each failed or
unexecuted condition's status, original error, traceback and memory evidence**,
including `COMPLETE_WITH_FAILURES`; no bare assertion hides the cause.
The existing 90 runner's partial-run exit convention remains unchanged; Slurm
must use this explicit validator.

## SERAPH launch (not executed locally)

```bash
cd /data/khuss/repos/GP_semcache
sbatch --export=ALL,R2C2_RUN_TAG=l1024_remaining jobs/r2c2_profile_remaining.sbatch
```

Adjust partition/resource directives with `sbatch` flags for the actual account.
The launcher defaults to `batch_eebme_ugrad`, one GPU, 4 CPUs, 48 GiB host RAM,
23:50:00; inspect these site-specific values before submitting. It uses the
existing offline Hugging Face environment and `conda activate semcache`.
It requires a Slurm GPU allocation, preserves `CUDA_VISIBLE_DEVICES`, and creates
`results/r2/c2/<tag>_<jobid>_<UTC timestamp>/` with a fail-on-collision root.

Within the **same physical GPU allocation**, in separate sequential processes:

1. Fresh Pilot, L512/L1024 × coverage 0/0.5/MAX; one warmup and three pairs.
2. L1024 coverage 0/MAX profile, guarded, three warmups.
3. L1024 coverage 0.5 profile, guarded, three warmups.

L512 profiling is **not repeated**. The launcher seals physical GPU UUID,
code hashes, branch/HEAD/dirty state and its own script hash in
`launch_identity.json`. It checks hashes before each run and checks each output's
code/GPU/profile completion afterward. The Slurm spool script is hashed as the
actual executed launcher. Pilot failure stops later runs with detailed evidence.

Manual equivalent inside a single allocated GPU session (all output paths new):

```bash
python scripts/90_run_r2c1_long_context_qkv.py --phase pilot \
  --model facebook/opt-2.7b --dtype float16 --device cuda:0 --cache-device cuda \
  --capacity-mib 512 --prompt-lengths 512,1024 --coverages 0,.5,max \
  --warmup-runs 1 --measured-runs 3 --seed 42 \
  --output-dir results/r2/c2/fresh_pilot

python scripts/90_run_r2c1_long_context_qkv.py --phase profile \
  --model facebook/opt-2.7b --dtype float16 --device cuda:0 --cache-device cuda \
  --capacity-mib 512 --prompt-lengths 1024 --coverages 0,max --seed 42 \
  --pilot-inputs results/r2/c2/fresh_pilot --profile-memory-policy pilot_guarded \
  --profile-memory-allowance-gib 1 --profile-memory-safety-gib 2 \
  --output-dir results/r2/c2/profile_1024_0_max

python scripts/90_run_r2c1_long_context_qkv.py --phase profile \
  --model facebook/opt-2.7b --dtype float16 --device cuda:0 --cache-device cuda \
  --capacity-mib 512 --prompt-lengths 1024 --coverages .5 --seed 42 \
  --pilot-inputs results/r2/c2/fresh_pilot --profile-memory-policy pilot_guarded \
  --profile-memory-allowance-gib 1 --profile-memory-safety-gib 2 \
  --output-dir results/r2/c2/profile_1024_half
```

Default trace budget is still 64 MiB. An actual oversized trace is preserved and
fails rather than changing workload. Set `R2C2_MAX_TRACE_MIB` before `sbatch` only
if an explicit larger host/disk trace budget is wanted; this does not change the
GPU memory policy or profiler instrumentation.

## Local validation and remaining limits

```bash
PYTHONPATH=src:tests python3 -m unittest test_r2c2_profile_memory \
  test_r2c1_workloads test_r2c1_benchmark test_r2c1_analysis -q
python3 scripts/90_run_r2c1_long_context_qkv.py --help
python3 scripts/91_validate_r2c2_results.py --help
bash -n jobs/r2c2_profile_remaining.sbatch
git diff --check
```

Tests use scalar/fake allocator and orchestration fixtures, not GPU findings.
Without local Torch/Transformers/PEFT, actual Pilot peak collection, GPU profile
completion, memory sufficiency and numeric parity remain **GPU_VALIDATION=NOT_RUN**.
No SERAPH job is submitted; no speedup or new capacity result is claimed. The
empirical guard reduces false conservative rejections but cannot guarantee absence
of fragmentation, external allocations, profiler growth or real CUDA OOM.

Local verification on 2026-10-28: 14 new focused tests passed. Combined R1,
R2-A, R2-B3.5, R2-C1 and R2-C2 regressions: 189 tests, 144 passed, 45
dependency-gated skips, zero failures. Python compilation, both CLI help commands,
all three launch-stage argument sets, `bash -n`, tracked diff whitespace and all
five new files' whitespace checks passed. Torch, Transformers and PEFT are absent
locally. **No actual GPU Pilot/profile or Slurm submission was performed.**
