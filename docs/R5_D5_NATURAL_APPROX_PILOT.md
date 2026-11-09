# D5 minimal existing-APPROX E2E pilot

Implementation and portable CPU fixtures: **PASS**. Native CPU approval and
real-asset preflight: **BLOCKED locally** (sealed SERAPH assets and
Torch/Transformers/NumPy are absent). GPU execution, physical NATURAL hits, task
degradation, TTFT and E2E: **NOT_MEASURED**. No GPU, Slurm, or push was executed.

The missing integration was in the NATURAL runner and auditor: script 153 pins
EXACT and two conditions, and scoped authorization replay expects exact-context
windows. The existing physical APPROX operator already consumes cached total
QKV and skips native projection and CPU LoRA rows. It is reused unchanged.

## Changes and scope

Script `157_r5_natural_approx.py` and `r5_natural_approx.py` add a separately
sealed `D5_NATURAL_APPROX_PILOT_V1` plan and one bounded run. Conditions execute
in fixed A/B/C order, each in a fresh process on the same GPU:

| Condition | Projection mode | Approximate authorization |
|---|---|---|
| FULL_RECOMPUTE | Full fresh prefill/decode | Disabled |
| SEMCACHE_D5_EXACT | EXACT_CONTEXT_VERIFIED | Disabled |
| SEMCACHE_D5_APPROX | APPROXIMATE_SUBSEQUENCE_TOTAL_QKV | Explicit opt-in |

`r5_subsequence_natural_runtime.py` adds optional runner/profile hooks; original
callers retain their defaults. Its ES–UD TCP, native generation, real TinyBERT,
per-adapter owners/indexes, CPU LoRA, actual attention Impact, completion ACK,
F/A/I/S, CHU, capacity, and PBR are reused. Owners persist for all requests in
one track/condition; they are disposed between conditions. No native target past
KV, attention, or FFN output is transferred from the source.

`r5_subsequence_natural_audit.py` accepts explicit condition/mode overrides and
an unscored controlled source. `r5_subsequence_audit.py` accepts non-prefix
scoped authorization coverage only under explicit approximate auditing. Source
completion, sharing, physical identity, admission proofs, authenticated private
snapshots, wire payloads, gather/add/scatter, and independent policy replay still
must pass. New pilot evidence requires its declared projection/integrity modes.
Default historical EXACT auditing is unchanged.

The physical operator, generation module, original pilot, integrity module,
sparse protocol, Phase 2.5A implementation, and historical artifacts are
unchanged. No new reuse algorithm or numerical thresholds were introduced.

Two separate tracks are available:

* **CONTROLLED:** a standalone three-token source from fixed positions `[32:35]`
  of original Block 0 request 0, then that original target. Source uses the same
  user/adapter, one fixed generated token, and is unscored. Both request costs
  are included. Preparation rejects a slice that is also the target prefix;
  there is no alternative selection. Real semantic assignment, admission,
  residency and authenticated consumption must succeed. This is an engineered
  correctness fixture, never a NATURAL observation.
* **NATURAL:** exactly original Block 0's five Qasper requests, with all original
  payloads, sample IDs, policy, routing and order preserved. Zero priming or
  inserted sources. No 150-request option or automatic expansion exists.

The controlled audit must report actual non-prefix consumption and projection
skipping before considering the NATURAL pilot. Zero NATURAL hits are valid;
retain the original workload and report zero.

## Evidence and interpretation

The new audit reuses independent source/value/wire/policy replay. It reports
consumed and skipped projection rows by layer/role, separately for A: same-position
shared prefix, B: same-position non-prefix, C: shifted non-prefix, and D: shifted
occurrences targeting the shared prefix. Counts sum projection rows across
layers and Q/K/V; they are not unique input-token counts.

Integrity PASS is separate from numerical tolerance, full-vocabulary logit
metrics/KL/argmax, bitwise logit agreement, generated token/text agreement, and
dataset scorer quality/delta versus Full Recompute. Thresholds are unchanged.
After generated-history divergence, logit comparisons are labeled diagnostic.
Approximate output differences never constitute an exact-equivalence PASS.
This five-request NATURAL pilot measures Qasper quality only.

Per-request TTFT, evidence-inclusive application E2E, total wire bytes, actual
computed/skipped rows, cache occupancy/policy and named cache/inference profiler
components are saved. Inclusive component scopes overlap; do not sum them.
Model initialization, final inventory sealing and timing sidecars are reported
separately. A fixed-order, single-block pilot with strict evidence overhead is
an instrumented observation, not a causal or statistical speedup estimate.
Reduced E2E without actual non-prefix skipping is not a non-prefix speedup;
changed generation lengths and quality must also be considered. The auditor
leaves correctness-qualified speedup `NOT_ESTABLISHED`.

## Resource preflight

Limits remain: input 512, generation 96, context 2048, 64 entries/candidates,
512 MiB/adapter cache, GPU forecast/allocated <22 GiB, ES 16 GiB/adapter,
UD 4 GiB/adapter, condition 32 GiB, run 64 GiB, 500,000 files, 96 GiB free disk,
32 GiB available host RAM, four CPUs/64 GiB allocation, 7,200 s per worker,
14,460 s shared A/B/C deadline, 4h30 allocation. The three workers share the
deadline; three separate 7,200 s allowances are not granted. Transport preflight
keeps 8 KiB headroom within the 64 KiB header limit.

Analytical envelopes below assume 512 input/32 maximum generated tokens for
each original target, and conservatively put all requests on one adapter. They
are fixtures for budgeting, not reconstructed real requests or measurements.

| Track | Requests/condition | GPU forecast | Evidence/condition | A/B/C evidence | Files including consumed snapshots |
|---|---:|---:|---:|---:|---:|
| CONTROLLED | 2 (3 + 512 input tokens) | 11.10 GiB | 2.59 GiB | 7.78 GiB | 72,780 |
| NATURAL | 5 | 11.10 GiB | 12.58 GiB | 37.74 GiB | 282,432 |

UD tensor payload alone forecasts 0.58/2.90 GiB per condition for those tracks;
real preparation additionally bounds transport/layer ledgers using actual
predicted headers and routing. The old ES writer checks its adapter parent,
including UD evidence. The original dense evidence forecast is retained.
Runtime also checks shared file/run bytes after each request and condition.
Planning at the old unmeasured 5 s/forward assumption gives 495/2,400 s for
A/B/C, excluding initialization and strict evidence overhead. These are not
runtime predictions supported by measurements. A five-by-512-by-96 three-condition
envelope fails the existing file guard. Never lower generation limits or raise
budgets to make preparation pass. PBR remains interval 100 per adapter and
cannot trigger in either bounded track.

## CPU results

`results/r5/i5r/d5_natural/approx_v1/cpu_local_20261109_02/` contains sealed
logs, source hashes, resource envelopes and the local receipt. New tests: 12
passed, 1 native test skipped. Existing D5 regressions: 84 passed, 8 native
tests skipped. Unchanged Phase 2.5A: 32 passed, 1 native test skipped. Total:
128 passed, 10 skipped, zero failures. The legacy test that executes a job shell
was excluded; the new job was syntax-checked with `bash -n` only.

Fixtures cover original first-five preservation, controlled provenance,
opt-ins, pinned plan reconstruction/tampering, bounded resources/deadline,
operator delegation, physical category accounting, numerical FAIL independent
of token agreement, and SCOPED non-prefix source-proof replay/corruption.
The native tiny-OPT CPU/TCP positive test is included but unavailable here.
No fixture result establishes OPT-2.7B behavior. GPU approval requires the
SERAPH `cpu-validate` command to complete with **zero skips**.

## SERAPH CPU preparation

Review and commit the listed implementation, shared hooks, CLI, job and tests
before preparation, then make that reviewed source available on SERAPH through
your normal transfer process. The existing source-identity guard requires a
clean, tracked implementation matching HEAD. No commit or push was performed
here. Use the existing offline `semcache` environment; do not download models
or replace sealed preparations.

```bash
cd /data/khuss/repos/GP_semcache
export I5_CONDA_ROOT=/absolute/path/to/existing/conda_root
source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=42
export D5NP_SOURCE_COMMIT="$(git rev-parse HEAD)"
export D5N_ORIGINAL="$PWD/results/r5/i5/plan_longbench3_smoke_v1"
export D5N_MULTI=/absolute/path/to/existing/sealed/expanded_30_user_preparation
export D5NP_ROOT="$PWD/results/r5/i5r/d5_natural/approx_v1"

CUDA_VISIBLE_DEVICES='' python scripts/157_r5_natural_approx.py prepare \
  --original-experiment "$D5N_ORIGINAL" --multi "$D5N_MULTI" \
  --source-commit "$D5NP_SOURCE_COMMIT" --track CONTROLLED \
  --enable-approximate-experiment --output-dir "$D5NP_ROOT/controlled_plan_01"
CUDA_VISIBLE_DEVICES='' python scripts/157_r5_natural_approx.py cpu-validate \
  --experiment "$D5NP_ROOT/controlled_plan_01" --output-dir "$D5NP_ROOT/controlled_cpu_01"
CUDA_VISIBLE_DEVICES='' python scripts/157_r5_natural_approx.py inspect \
  --experiment "$D5NP_ROOT/controlled_plan_01" --cpu-audit "$D5NP_ROOT/controlled_cpu_01"

CUDA_VISIBLE_DEVICES='' python scripts/157_r5_natural_approx.py prepare \
  --original-experiment "$D5N_ORIGINAL" --multi "$D5N_MULTI" \
  --source-commit "$D5NP_SOURCE_COMMIT" --track NATURAL \
  --enable-approximate-experiment --output-dir "$D5NP_ROOT/natural_plan_01"
CUDA_VISIBLE_DEVICES='' python scripts/157_r5_natural_approx.py cpu-validate \
  --experiment "$D5NP_ROOT/natural_plan_01" --output-dir "$D5NP_ROOT/natural_cpu_01"
CUDA_VISIBLE_DEVICES='' python scripts/157_r5_natural_approx.py inspect \
  --experiment "$D5NP_ROOT/natural_plan_01" --cpu-audit "$D5NP_ROOT/natural_cpu_01"
```

Resolve `D5N_MULTI` from the existing D5 NATURAL `natural_plan.json` field
`multi`; `original_experiment` records the original sealed workload. All outputs
above must be fresh. Missing/invalid assets, changed source, or resource guards
stop preparation. Review actual resource forecasts before requesting a pilot.

## Subsequent manually approved pilot only

The following instructions are for a later approved GPU action; they were
**not executed**. Run one track per allocation. First review CONTROLLED CPU
approval and request approval for this bounded job:

```bash
export D5NP_EXPERIMENT="$D5NP_ROOT/controlled_plan_01"
export D5NP_CPU_AUDIT="$D5NP_ROOT/controlled_cpu_01"
export D5NP_RUN_ROOT="$D5NP_ROOT/controlled_attempts"
export D5N_NODE=moana-y3
export R5_ENABLE_GPU=1 R5_I5_ENABLE_GPU=1
mkdir -p logs "$D5NP_RUN_ROOT"
sbatch --nodelist="$D5N_NODE" --export=ALL jobs/r5_natural_approx.sh
```

After completion, substitute the returned job ID; audit entirely on CPU:

```bash
export D5NP_RUN="$D5NP_RUN_ROOT/run_RETURNED_JOB_ID"
CUDA_VISIBLE_DEVICES='' python scripts/157_r5_natural_approx.py analyze \
  --experiment "$D5NP_EXPERIMENT" --cpu-audit "$D5NP_CPU_AUDIT" \
  --run "$D5NP_RUN" --output-dir "${D5NP_RUN}_audit"
CUDA_VISIBLE_DEVICES='' python scripts/157_r5_natural_approx.py verify \
  --experiment "$D5NP_EXPERIMENT" --cpu-audit "$D5NP_CPU_AUDIT" \
  --run "$D5NP_RUN" --audit "${D5NP_RUN}_audit" --output-dir "${D5NP_RUN}_verification"
```

Inspect `controlled_nonprefix_consumption=PASS`, nonzero B/C consumed/skipped
rows, integrity PASS, numerical/quality differences and costs. If the controlled
mechanism fails, report the failure before any NATURAL run. After a separate
manual decision, use the same job/audit commands with `natural_plan_01`,
`natural_cpu_01`, and fresh `natural_attempts`. Do not chain submissions or
expand to 150 requests. A NATURAL zero-hit result remains a valid result.
