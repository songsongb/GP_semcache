#!/usr/bin/env bash
#SBATCH --job-name=r5_i5r3bbe
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_i5r3bbe_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5_i5r3bbe_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?manual GPU allocation required}" "${IBE_EXPERIMENT:?sealed plan required}"
: "${IBE_SOURCE_COMMIT:?reviewed source commit required}" "${IBE_RUN_ROOT:?fresh attempt parent required}"
: "${IBE_NODE:?select moana-y3 or moana-y4}" "${I5_CONDA_ROOT:?Conda base path required}"
[[ "$IBE_NODE" == moana-y3 || "$IBE_NODE" == moana-y4 ]] || exit 2
[[ "${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-}}" == "$IBE_NODE" ]] || exit 2
[[ "${R5_I5_ENABLE_GPU:-0}" == 1 && "${R5_ENABLE_GPU:-0}" == 1 && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo 'One explicit non-array smoke/group allocation required' >&2; exit 2; }
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration && "$(git rev-parse HEAD)" == "$IBE_SOURCE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || exit 2
source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$IBE_RUN_ROOT"
python -u scripts/146_r5_i5r3bbe.py run --experiment "$IBE_EXPERIMENT" \
 --block-index "${IBE_BLOCK_INDEX:-0}" --source-commit "$IBE_SOURCE_COMMIT" \
 --output-dir "$IBE_RUN_ROOT/block_${IBE_BLOCK_INDEX:-0}_${SLURM_JOB_ID}" \
 --enable-gpu-experiment
