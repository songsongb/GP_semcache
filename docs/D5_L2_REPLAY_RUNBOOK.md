# L2 continuous replay: CPU preflight and manual GPU gates

This revision implements replay; **local approval is PARTIAL_BLOCKED** because
this Ubuntu host has no native packages or real prepared assets and `/data/khuss`
is absent. No GPU/Slurm job was run/submitted. Actual dataset counts and first-five
matches must be checked on SERAPH. See `D5_L2_REPLAY_CPU_REPORT.md` for evidence.

## Contract and compatibility

Default `prepare` remains NATURAL: duplicate original samples are forbidden.
Replay requires both `--workload-mode CONTINUOUS_REPLAY` and
`--enable-continuous-replay`. It validates every original repetition-0 NATURAL
block and traverses **all** its requests in original order before cycling. There
is no HIT/length/adapter-based selection. First-cycle tasks remain byte-identical;
only repeated occurrences receive `D5REPLAY_<ordinal>_<source-hash>` execution IDs.
All execution IDs are unique. Original sequence numbers stay inside each request;
global session ordinals are separate. Sample IDs, prompt, tokens, generation,
seed, user and adapter routing are unchanged.

Parallel occurrence metadata binds original request/block IDs and indices, pool
position, complete original task SHA256, cycle and occurrence index (zero-based).
The plan reports unique samples/original requests, repeats/ratio, complete/partial
cycles and user/adapter distributions. Partial cycles cannot split groups. The
source pool must itself be duplicate-free and valid; replay does not repair it.
This is a **new continuous session**, not evidence that original cold blocks
were continuous. The legacy continuous-contract JSON option is not used here.

`--baseline-plan` references a frozen same-dataset NATURAL Light five-request
plan. Its inventory is pinned and its first five full tasks/seed/input identity
and model-adapter assets are compared. A mismatch records NOT_MATCHED and reasons;
missing baseline also records NOT_MATCHED. MATCHED_SELECTION_ONLY is not native
fidelity or statistical performance reproduction. Frozen baseline files/approvals
are read-only and never reused as approval of this changed source.

Within a condition, one ES/model and one owner/index/CPU UD per adapter persist.
Only the adapter's final authenticated completion ACK permits its STOP/BYE,
before potentially long local admission; owners remain until condition cleanup.
Conditions start fresh. No cache checkpoint, split job, rerouting or cross-adapter
reuse is introduced. Original proof/clone/hash, w=3, QKV/LoRA, attention impact,
F/A/I/S, admission/eviction/CHU/PBR and wire arithmetic are unchanged.

Replay adds bounded occurrence metadata copying/validation and slightly larger
compact receipt persistence inside the existing application E2E interval. This
observer cost is included, not subtracted; model TTFT/prefill/decode scope
definitions remain unchanged. NATURAL requests do not get this extra metadata.

## User CPU preparation on SERAPH after review/commit/push

Use the dedicated Light worktree and already installed `semcache` environment.
Do not switch the Strict tree or install/download anything in the job. All new
outputs, temporary files and caches must be under `/data/khuss`. The current
working source is dirty/uncommitted, so a production plan is intentionally blocked
until the user reviews/commits it. No commit or push was performed by Codex.

```bash
set -euo pipefail
cd /data/khuss/repos/GP_semcache_lightperf
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
: "${D5L_TMPDIR:?new/existing approved scratch directory under /data/khuss}"
export TMPDIR="$D5L_TMPDIR" TORCH_HOME="$D5L_TMPDIR/torch" CUDA_CACHE_PATH="$D5L_TMPDIR/cuda_cache"
python3 scripts/159_r5_d5_lightperf.py environment
python3 scripts/159_r5_d5_lightperf.py fingerprint
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest \
  test_r5_d5_lightperf test_r5_d5_lightperf_worker test_r5_d5_lightperf_replay \
  test_r5_d5_lightperf_native -v
```

Select one tuple with verified existing original I5/multi/base/tokenizer/TinyBERT/
dataset/scorer assets. Repeat CPU preparation for Qasper, 2WikiMQA, MultiNews at
20 and 50 after inspecting each report; 5 is supported too. Do not assume all
datasets use the same I5 plan or have ten source requests.

```bash
: "${D5_REVIEWED_COMMIT:?new clean reviewed commit}"
: "${D5_REVIEWED_FINGERPRINT:?new fingerprint from that committed source}"
: "${D5_DATASET:?qasper or 2wikimqa or multi_news}"
: "${D5_COUNT:?20 or 50}"
: "${D5_ORIGINAL_I5:?existing original dataset-specific I5 plan}"
: "${D5_MULTI:?existing expanded multi-adapter preparation}"
: "${D5_BASELINE_5:?frozen NATURAL Light five-request plan for this dataset}"
: "${D5_TAG:?new unique tag}"
D5_ROOT=/data/khuss/repos/GP_semcache_lightperf/results/r5/d5_lightperf
D5_PLAN="$D5_ROOT/${D5_DATASET}_${D5_COUNT}_replay_plan_$D5_TAG"
D5_CPU="$D5_ROOT/${D5_DATASET}_${D5_COUNT}_replay_cpu_$D5_TAG"
python3 scripts/159_r5_d5_lightperf.py prepare \
  --dataset "$D5_DATASET" --count "$D5_COUNT" --original-experiment "$D5_ORIGINAL_I5" --multi "$D5_MULTI" \
  --workload-mode CONTINUOUS_REPLAY --enable-continuous-replay --baseline-plan "$D5_BASELINE_5" \
  --reviewed-commit "$D5_REVIEWED_COMMIT" --reviewed-fingerprint "$D5_REVIEWED_FINGERPRINT" --output-dir "$D5_PLAN"
python3 scripts/159_r5_d5_lightperf.py preflight --experiment "$D5_PLAN"
python3 scripts/159_r5_d5_lightperf.py cpu-validate --experiment "$D5_PLAN" --output-dir "$D5_CPU"
python3 scripts/159_r5_d5_lightperf.py inspect --experiment "$D5_PLAN" --cpu-audit "$D5_CPU"
```

`preflight` emits JSON to stdout: actual request list/hash, original pool count,
unique/repeated samples and ratio, user/adapter counts, first-five match, context/
forward/exchange/wire/RAM/VRAM/cache/disk/file/time bounds and reasons. PASS means
static bounds only; `gpu_execution_approved=false`. BLOCKED_INFEASIBLE exits2.
Missing source/assets/packages fail closed. Output may be redirected by the user
to a **fresh** path under `/data/khuss`; never overwrite a saved report.

The old 7200s worker /16200s Slurm ceiling, 22GiB allocation ceiling, post-load
12GiB reserve, ≥4 CPUs, 64GiB allocation/≥32GiB available host memory, cache512MiB
per adapter, input512/context2048/output96 and IO limits remain unchanged. No HIT
savings are assumed. 5s/maximum forward +600s initialization reserve is unmeasured
planning, not a fit to the five-request GPU result. A blocked tuple stays blocked;
do not change tokens, generation, deadlines or resources to force passage. Large
repeated plan JSON can also hit the existing8MiB cap; no metadata truncation occurs.

Replay approvals additionally require `test_r5_d5_lightperf_replay`; native tests
include real Tiny OPT replay at20/50 across all three conditions, comparing RAM
logits with Strict and checking persistent cache/route/STOP/BYE. Missing native
packages fail, not skip. All required native/regression tests must pass with zero
failures/errors/skips before a clean CPU approval is issued. NATURAL approval
module lists and offline historical five-request verification remain compatible.

## Future manually approved GPU sequence — not executed in this task

Use `jobs/r5_d5_lightperf_replay.sh`, one job per dataset/scale/condition. It rejects
arrays, missing approvals/opt-ins, non-20/50 replay, non-baseline knobs and paths
outside `/data/khuss`. Original `jobs/r5_d5_lightperf.sh` remains unchanged.
Current replay jobs use unlimited bandwidth/factor1. Generic bandwidth/factor
interfaces remain for future separate sensitivity approval; no sweep is submitted.

After CPU approval and separate user authorization, select the actual allocated
RTX3090 UUID and approved node; preserve real CUDA visibility in the GPU job.
Create the log directory under `/data/khuss` manually if absent. User supplies
remaining allocation seconds; no Slurm query is performed by these scripts.

```bash
export D5L_REPO_ROOT=/data/khuss/repos/GP_semcache_lightperf
export D5L_EXPERIMENT="$D5_PLAN" D5L_CPU_AUDIT="$D5_CPU" D5L_SOURCE_COMMIT="$D5_REVIEWED_COMMIT"
export D5L_ENABLE_GPU=1 R5_ENABLE_GPU=1 R5_I5_ENABLE_GPU=1 D5L_ENABLE_EXTENDED=1 D5L_ENABLE_REPLAY=1
export D5N_NODE=moana-y5
: "${D5L_GPU_UUID:?actual expected GPU UUID}"
: "${D5L_ALLOCATION_REMAINING_SECONDS:?actual remaining time <=16200}"
export D5L_GPU_UUID D5L_ALLOCATION_REMAINING_SECONDS D5L_TMPDIR HF_HOME
export D5L_CONDITION=FULL_RECOMPUTE
export D5L_OUTPUT="$D5_ROOT/${D5_DATASET}_${D5_COUNT}_replay_full_$D5_TAG"
# Let Slurm supply the allocated GPU visibility rather than inheriting the CPU-only shell setting.
unset CUDA_VISIBLE_DEVICES
# User only, after explicit GPU approval; never submit all combinations.
sbatch --nodelist="$D5N_NODE" jobs/r5_d5_lightperf_replay.sh
```

After completion, verify FULL before manually choosing EXACT; then verify EXACT
before APPROX. APPROX needs `D5L_ENABLE_APPROX=1`. Each uses a new D5L_OUTPUT;
never resubmit a failed condition automatically. Keep the RUN and all sibling
RUN__ud_Adapter_* receipts together and immutable.

```bash
CUDA_VISIBLE_DEVICES='' python3 scripts/159_r5_d5_lightperf.py verify \
  --experiment "$D5_PLAN" --cpu-audit "$D5_CPU" --run "$D5L_OUTPUT"
# After each approved condition, extend an explicit registry of {run, plan, cpu_audit} triples.
: "${D5_RESULT_REGISTRY:?verified result registry under /data/khuss}"
CUDA_VISIBLE_DEVICES='' python3 scripts/159_r5_d5_lightperf.py analyze \
  --registry "$D5_RESULT_REGISTRY" --output-dir "$D5_ROOT/${D5_DATASET}_${D5_COUNT}_replay_analysis_$D5_TAG"
```

Outputs retain analysis.json/tables.csv/tables.md and add position_analysis.json
and position_windows.csv. They contain mean/median/P95 TTFT/E2E, request/token
throughput and matched FULL comparisons, per-user/adapter HIT/reuse, QKV skip
ratio, occupancy/policy counts, unique/repeated requests and five-request windows.
Completion HIT/MISS splits and raw phase/component bottlenecks remain wall scopes.
Missing measurements are null/NOT_MEASURED; inclusive scopes overlap and are not
added into E2E. Window token/output/user/adapter mix is reported, so declining
cost is not automatically attributed to warm-up. P95 in five-request bins is
exploratory. Task quality is never pooled across datasets. Measured speed ratios
and token agreement are descriptive; no threshold is extrapolated. No observed
equality is NO_OBSERVED_BREAK_EVEN, not a fabricated hardware break-even point.

After analyzing one scale, the user decides whether any next scale is feasible
and worth running. Preserve correctness_qualified_speedup=NOT_ESTABLISHED and
independent numerical audit NOT_EVALUATED. Neither CPU fixtures nor replay HITs
establish GPU speedup, fidelity or true remote-network/CPU hardware equivalence.
