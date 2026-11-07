#!/usr/bin/env bash
#SBATCH --job-name=r5_i5r_pilot
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:20:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_i5r_pilot_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5_i5r_pilot_%j.err
set -Eeuo pipefail
: "${SLURM_JOB_ID:?manual allocation required}" "${IP_EXPERIMENT:?sealed plan required}"
: "${IP_SOURCE_COMMIT:?reviewed source required}" "${IP_RUN_ROOT:?fresh run parent required}"
: "${IP_NODE:?select moana-y3 or moana-y4}" "${I5_CONDA_ROOT:?conda root required}"
[[ "$IP_NODE" == moana-y3 || "$IP_NODE" == moana-y4 ]] || exit 2
[[ "${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-}}" == "$IP_NODE" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 2
[[ "${R5_I5_ENABLE_GPU:-0}" == 1 && "${R5_ENABLE_GPU:-0}" == 1 ]] || exit 2
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration && "$(git rev-parse HEAD)" == "$IP_SOURCE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || exit 2
[[ -d logs && -d "$IP_RUN_ROOT" ]] || { echo 'logs and run parent must exist before submission' >&2; exit 2; }
source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=42
timeout --signal=TERM --kill-after=30s 6h python -u scripts/149_r5_i5r_parallel.py run \
 --experiment "$IP_EXPERIMENT" --source-commit "$IP_SOURCE_COMMIT" \
 --block-index "${IP_BLOCK_INDEX:-0}" --output-dir "$IP_RUN_ROOT/block_${IP_BLOCK_INDEX:-0}_${SLURM_JOB_ID}" --enable-gpu-experiment
