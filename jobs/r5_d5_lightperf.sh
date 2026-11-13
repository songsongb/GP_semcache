#!/bin/bash
# Manual template only. No submission has been performed.
#SBATCH --job-name=d5-light-manual
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:30:00
set -euo pipefail
# No invocation/submission is performed by Codex. Every variable is supplied by
# the user after clean committed native CPU approval on the execution host.
: "${D5L_REPO_ROOT:?explicit Light worktree}"
: "${D5L_EXPERIMENT:?fresh sealed real plan}"
: "${D5L_CPU_AUDIT:?fresh zero-skip native Light CPU approval}"
: "${D5L_SOURCE_COMMIT:?reviewed clean Light commit}"
: "${D5L_GPU_UUID:?expected physical GPU UUID}"
: "${D5L_CONDITION:?one explicit condition}"
: "${D5L_OUTPUT:?fresh condition output}"
: "${D5L_ALLOCATION_REMAINING_SECONDS:?explicit remaining allocation time}"
[[ "${D5L_ENABLE_GPU:-0}" == 1 && "${R5_ENABLE_GPU:-0}" == 1 && "${R5_I5_ENABLE_GPU:-0}" == 1 ]] || exit 2
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 2
cd "$D5L_REPO_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=42
CUDA_VISIBLE_DEVICES='' python3 scripts/159_r5_d5_lightperf.py inspect \
  --experiment "$D5L_EXPERIMENT" --cpu-audit "$D5L_CPU_AUDIT"
D5L_EXTRA_ARGS=()
if [[ "$D5L_CONDITION" == SEMCACHE_D5_APPROX ]]; then
  [[ "${D5L_ENABLE_APPROX:-0}" == 1 ]] || exit 2
  D5L_EXTRA_ARGS+=(--enable-approximate-experiment)
fi
# Baseline 5-request only. Extended jobs require a separately reviewed manual
# command with --enable-extended-experiment; this template does not expand a grid.
python3 -u scripts/159_r5_d5_lightperf.py run \
  --experiment "$D5L_EXPERIMENT" --cpu-audit "$D5L_CPU_AUDIT" \
  --condition "$D5L_CONDITION" --source-commit "$D5L_SOURCE_COMMIT" \
  --expected-gpu-uuid "$D5L_GPU_UUID" --output-dir "$D5L_OUTPUT" \
  --enable-gpu-experiment "${D5L_EXTRA_ARGS[@]}"
