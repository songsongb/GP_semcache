# Native D5 Light: CPU approval and later manual GPU pilot

Current local status: **PARTIAL_BLOCKED**. Native ES/UD orchestration and mandatory
Tiny OPT CPU tests are implemented, but torch/transformers/numpy/rouge and real
prepared assets are absent here. No native parity PASS or GPU approval has been
issued. The commands below are for the user after source review/commit/push and
transfer to the actual SERAPH Light worktree. No GPU/Slurm action is part of this task.

## Source and offline assets

Use `exp/d5-l1-l5-lightperf` in a dedicated worktree. This follow-up started at
`6708d369f42577a3fbe8b46411aa98cdbac4b776`, containing Strict baseline
`4916836ce328e119f3a8d85c6c1afc7cf8d363e5`. Follow-up edits are uncommitted until
the user reviews/authorizes a commit. Do not modify the original Strict worktree.
Every source/root/commit/input change needs new PLAN and CPU audit; old approvals
are not reusable. Production rejects development source and TEST_FIXTURE_ONLY assets.

Use an existing offline environment matching prepared versions; install nothing
as part of this task. Required assets:

* Original sealed I5 workload/dataset manifests and real Qasper/2WikiMQA/MultiNews
  prepared records, preserving exact prompts, routing, seed, IDs and generation.
* Existing OPT-2.7B FP16 base/tokenizer snapshots at immutable revisions, expanded
  30-user/three-adapter registry and distinct rank8 FP32 A/B matrices.
* Pinned local TinyBERT checkpoint/tokenizer; CPU prepare loads and fingerprints
  it. MultiNews requires the existing `rouge==1.0.1` scorer.
* Each dataset's frozen NATURAL five-request Strict plan for matched selection.
  Without complete first-five task equality, selection is NOT_MATCHED. Selection
  equality alone is not native parity or GPU fidelity evidence.

Actual SERAPH paths must be supplied by the user; they are not inferred from
this Ubuntu host. Do not rebuild data, shrink generation or raise budgets to pass.

## CPU stage after user push

Activate the already provisioned native environment in the reviewed Light worktree.
These commands do not query or submit Slurm jobs.

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
pwd
git branch --show-current
git rev-parse HEAD
git status --short
python3 scripts/159_r5_d5_lightperf.py environment
python3 scripts/159_r5_d5_lightperf.py fingerprint
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest \
  test_r5_d5_lightperf test_r5_d5_lightperf_worker -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest test_r5_d5_lightperf_native -v
```

Native tests create Tiny OPT from config, real CPU tensor LoRA and spawned UDs;
they download nothing. They compare full-vocabulary logits in memory against the
unchanged Strict generator, plus tokens, real impact, owner integrity, row skips,
policy/eviction, routing, decode/EOS and final ACK/STOP/BYE. Rate/factor variants
exercise the same arithmetic. These are synthetic native correctness tests, not
real dataset quality/GPU proof. Missing packages FAIL instead of silently skipping.

Assign the **new clean reviewed** commit/fingerprint and verified local paths.
Different datasets may use different original I5 plans. D5_TAG must be new.

```bash
: "${D5_REVIEWED_COMMIT:?reviewed clean new Light commit}"
: "${D5_REVIEWED_FINGERPRINT:?fresh content fingerprint}"
: "${D5_QASPER_I5:?existing original Qasper I5 plan}"
: "${D5_2WIKI_I5:?existing original 2WikiMQA I5 plan}"
: "${D5_MULTINEWS_I5:?existing original MultiNews I5 plan}"
: "${D5_QASPER_STRICT_PLAN:?frozen Qasper five-request plan}"
: "${D5_2WIKI_STRICT_PLAN:?frozen 2WikiMQA five-request plan}"
: "${D5_MULTINEWS_STRICT_PLAN:?frozen MultiNews five-request plan}"
: "${D5_MULTI:?expanded multi-adapter preparation}"
: "${D5_TAG:?new unique tag}"
D5_OUTPUT_ROOT="$PWD/results/r5/d5_lightperf"

python3 scripts/159_r5_d5_lightperf.py prepare --dataset qasper --count 5 \
  --original-experiment "$D5_QASPER_I5" --multi "$D5_MULTI" --strict-pilot "$D5_QASPER_STRICT_PLAN" \
  --reviewed-commit "$D5_REVIEWED_COMMIT" --reviewed-fingerprint "$D5_REVIEWED_FINGERPRINT" \
  --output-dir "$D5_OUTPUT_ROOT/qasper_5_plan_$D5_TAG"
python3 scripts/159_r5_d5_lightperf.py prepare --dataset 2wikimqa --count 5 \
  --original-experiment "$D5_2WIKI_I5" --multi "$D5_MULTI" --strict-pilot "$D5_2WIKI_STRICT_PLAN" \
  --reviewed-commit "$D5_REVIEWED_COMMIT" --reviewed-fingerprint "$D5_REVIEWED_FINGERPRINT" \
  --output-dir "$D5_OUTPUT_ROOT/2wikimqa_5_plan_$D5_TAG"
python3 scripts/159_r5_d5_lightperf.py prepare --dataset multi_news --count 5 \
  --original-experiment "$D5_MULTINEWS_I5" --multi "$D5_MULTI" --strict-pilot "$D5_MULTINEWS_STRICT_PLAN" \
  --reviewed-commit "$D5_REVIEWED_COMMIT" --reviewed-fingerprint "$D5_REVIEWED_FINGERPRINT" \
  --output-dir "$D5_OUTPUT_ROOT/multi_news_5_plan_$D5_TAG"

for D5_DATASET in qasper 2wikimqa multi_news; do
  python3 scripts/159_r5_d5_lightperf.py cpu-validate \
    --experiment "$D5_OUTPUT_ROOT/${D5_DATASET}_5_plan_$D5_TAG" \
    --output-dir "$D5_OUTPUT_ROOT/${D5_DATASET}_5_cpu_$D5_TAG"
  python3 scripts/159_r5_d5_lightperf.py inspect \
    --experiment "$D5_OUTPUT_ROOT/${D5_DATASET}_5_plan_$D5_TAG" \
    --cpu-audit "$D5_OUTPUT_ROOT/${D5_DATASET}_5_cpu_$D5_TAG"
done
```

CPU approval requires clean reviewed source, real input reconstruction before and
after tests, engine D5_LIGHT_NATIVE_V2, installed versions and all required native
Light/existing regressions with zero failures/errors/skips and no CUDA context.
Only then is CLEAN_NATIVE_CPU_APPROVAL issued. Development source may receive
DEVELOPMENT_TESTS_PASS, never GPU permission. Old engine/Strict/scalar-only audits
are rejected. `inspect` can report CPU_APPROVED_MANUAL_GPU_NOT_RUN only with a
fresh valid audit and feasible resources. Source changes invalidate the fingerprint.

## Later user-approved five-request GPU command

**Do not execute this stage in the present CPU-only task.** After all CPU gates
and separate user GPU approval, select one condition in a manual non-array
allocation. Keep actual GPU UUID/model/backend/precision/CPU policy identical.

Required: one compatible original 24GiB-class GPU, <22GiB actual peak and ≥12GiB
free after loading; ≥4 CPUs; 64GiB allocation and ≥32GiB available host/cgroup RAM;
≥1GiB free disk. Template is 04:30:00, worker deadline remains ≤7200s, and parent
watchdog includes startup. User provides remaining allocation seconds without an
automatic Slurm query. At 512 input/original caps and **unmeasured** 5s/forward,
five-request forecasts are ~1400s/condition for Qasper/2Wiki and ~3000s for MultiNews.
These estimates do not justify extending limits.

Within the authorized allocation, preserve its real CUDA visibility: the CPU stage
above hid it. Select the actual GPU UUID and allocated allowed node; do not invent
values. No srun/sbatch command has been executed in this task.

```bash
: "${D5L_GPU_UUID:?actual physical GPU UUID}"
: "${D5N_NODE:?moana-y1/y2/y3/y4/y5/y7 matching allocated node}"
: "${D5L_ALLOCATION_REMAINING_SECONDS:?actual remaining seconds <=16200}"
export D5L_ENABLE_GPU=1 R5_ENABLE_GPU=1 R5_I5_ENABLE_GPU=1
python3 scripts/159_r5_d5_lightperf.py run \
  --experiment "$D5_OUTPUT_ROOT/qasper_5_plan_$D5_TAG" \
  --cpu-audit "$D5_OUTPUT_ROOT/qasper_5_cpu_$D5_TAG" \
  --source-commit "$D5_REVIEWED_COMMIT" --expected-gpu-uuid "$D5L_GPU_UUID" \
  --condition FULL_RECOMPUTE --enable-gpu-experiment \
  --output-dir "$D5_OUTPUT_ROOT/qasper_5_full_$D5_TAG"
```

EXACT uses `--condition SEMCACHE_D5_EXACT` and a fresh exact output. APPROX uses
`--condition SEMCACHE_D5_APPROX --enable-approximate-experiment` and a fresh approx
output. Use analogous dataset-specific plans for 2WikiMQA/MultiNews; each is a
separate manual decision. The job template `jobs/r5_d5_lightperf.sh` is an optional
future user-configured wrapper: it first runs CPU inspect and rejects missing
source/approval/assets/allocation/opt-ins before GPU setup. It submits nothing.

Keep RUN and fresh `RUN__ud_Adapter_A/B/C` sibling directories immutable together.
Their seals and ordered wire/service summaries are bound to run.json. Any
failure.json invalidates success. There is no retry/fallback or Strict-as-Light path.

```bash
CUDA_VISIBLE_DEVICES='' python3 scripts/159_r5_d5_lightperf.py verify \
  --experiment "$D5_OUTPUT_ROOT/qasper_5_plan_$D5_TAG" \
  --cpu-audit "$D5_OUTPUT_ROOT/qasper_5_cpu_$D5_TAG" \
  --run "$D5_OUTPUT_ROOT/qasper_5_full_$D5_TAG"
# Registry: explicit JSON list of {run, plan, cpu_audit} triples, no duplicates.
CUDA_VISIBLE_DEVICES='' python3 scripts/159_r5_d5_lightperf.py analyze \
  --registry "$D5_RESULT_REGISTRY" --output-dir "$D5_OUTPUT_ROOT/analysis_$D5_TAG"
```

Exports: analysis.json, tables.csv, tables.md and Light seal. Model/QKV components
are wall-only; GPU kernel, driver, queue, total observer and final seal durations
are NOT_MEASURED. Inclusive scopes overlap and cannot reconstruct E2E by summing.
Quality and primary persistence are separately measured inside application latency;
final summary/sealing are excluded. Link achieved Mbps include inter-frame gaps;
this is not a genuine remote link. UD ratios are measured per-operation service
emulation, never hardware-equivalent CPU speed.

## 20/50 and L3/L4/L5 remain separate gates

Original prefix/routing/group checks remain for 5/20/50. Cross-block streams need
an explicit new continuous identity contract, with cache retained for the whole
condition. Static infeasibility remains blocked. Conservative Qasper/2Wiki50 and
MultiNews20/50 exceed the existing 7200s bound; do not shrink generation or raise
limits. Each condition is one continuous session, never split across jobs.

Bandwidth choices: omitted/unlimited, 200/100/50 Mbps, both directions per channel.
UD factors: 1/.5/.25, EMULATED_UD_SERVICE_TIME. Knobs are fixed in PLAN and apply
to every condition. Non-baseline knobs/counts require a separate explicit
`--enable-extended-experiment` after evidence and user approval; the baseline job
template does not supply it. RTT/shared-link scheduling are unsupported. Matrix
still submits zero jobs; targeted L5 subsets only after separate L3/L4 evidence.
Plot export and automatic measured-crossing series grouping remain outside this revision.

Keep correctness_qualified_speedup=NOT_ESTABLISHED and persisted independent
full-vocabulary audit NOT_EVALUATED. Historical Strict MultiNews 221601 files /
34.866GB, EXACT 2/5 HIT/2880 rows/fidelity5/5, APPROX 2/5 HIT/10944 rows/fidelity3/5
are references, not PASS results for the new native Light implementation.
