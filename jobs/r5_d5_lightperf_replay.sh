#!/usr/bin/env bash
# Manual submission only: exactly one dataset/20-or-50/condition, no arrays/sweep.
#SBATCH --job-name=d5-replay-manual
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:30:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache_lightperf
#SBATCH --output=/data/khuss/repos/GP_semcache_lightperf/logs/d5_replay_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache_lightperf/logs/d5_replay_%j.err
set -euo pipefail
: "${D5L_REPO_ROOT:?reviewed Light worktree under /data/khuss}"
: "${D5L_EXPERIMENT:?fresh continuous replay plan}"
: "${D5L_CPU_AUDIT:?new native zero-skip CPU approval}"
: "${D5L_SOURCE_COMMIT:?reviewed clean commit}"
: "${D5L_GPU_UUID:?actual expected GPU UUID}"
: "${D5L_CONDITION:?one explicit FULL/EXACT/APPROX condition}"
: "${D5L_OUTPUT:?fresh condition output under /data/khuss}"
: "${D5N_NODE:?explicit approved allocated node}"
: "${D5L_ALLOCATION_REMAINING_SECONDS:?actual remaining allocation time <=16200}"
: "${D5L_TMPDIR:?existing temporary directory under /data/khuss}"
: "${HF_HOME:?existing offline HF cache under /data/khuss}"
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 2
[[ "${D5L_ENABLE_GPU:-0}" == 1 && "${R5_ENABLE_GPU:-0}" == 1 && "${R5_I5_ENABLE_GPU:-0}" == 1 ]] || exit 2
[[ "${D5L_ENABLE_EXTENDED:-0}" == 1 && "${D5L_ENABLE_REPLAY:-0}" == 1 ]] || exit 2
for D5L_PATH in "$D5L_REPO_ROOT" "$D5L_EXPERIMENT" "$D5L_CPU_AUDIT" "$D5L_OUTPUT" "$D5L_TMPDIR" "$HF_HOME"; do
  case "$(realpath -m -- "$D5L_PATH")" in /data/khuss/*) ;; *) exit 2 ;; esac
done
cd "$D5L_REPO_ROOT"
[[ -d "$D5L_TMPDIR" && -d "$HF_HOME" ]] || exit 2
export TMPDIR="$D5L_TMPDIR" HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$D5L_TMPDIR/torch" CUDA_CACHE_PATH="$D5L_TMPDIR/cuda_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=42
CUDA_VISIBLE_DEVICES='' python3 - "$D5L_EXPERIMENT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from semcache.edgelora.communication import require
from semcache.edgelora.r5_d5_lightperf import load_plan
from semcache.edgelora.r5_d5_lightperf_replay import replay_enabled
from semcache.edgelora.r5_d5_lightperf_workload import knobs
p, _ = load_plan(sys.argv[1])
require(replay_enabled(p) and p['count'] in (20, 50) and p['performance_knobs'] == knobs(),
        'extended template requires 20/50 CONTINUOUS_REPLAY, unlimited/factor1')
PY
CUDA_VISIBLE_DEVICES='' python3 scripts/159_r5_d5_lightperf.py preflight --experiment "$D5L_EXPERIMENT"
CUDA_VISIBLE_DEVICES='' python3 scripts/159_r5_d5_lightperf.py inspect \
  --experiment "$D5L_EXPERIMENT" --cpu-audit "$D5L_CPU_AUDIT"
D5L_EXTRA_ARGS=()
case "$D5L_CONDITION" in
  FULL_RECOMPUTE|SEMCACHE_D5_EXACT) ;;
  SEMCACHE_D5_APPROX)
    [[ "${D5L_ENABLE_APPROX:-0}" == 1 ]] || exit 2
    D5L_EXTRA_ARGS+=(--enable-approximate-experiment) ;;
  *) exit 2 ;;
esac
python3 -u scripts/159_r5_d5_lightperf.py run \
  --experiment "$D5L_EXPERIMENT" --cpu-audit "$D5L_CPU_AUDIT" \
  --condition "$D5L_CONDITION" --source-commit "$D5L_SOURCE_COMMIT" \
  --expected-gpu-uuid "$D5L_GPU_UUID" --output-dir "$D5L_OUTPUT" \
  --enable-gpu-experiment --enable-extended-experiment --enable-continuous-replay "${D5L_EXTRA_ARGS[@]}"
