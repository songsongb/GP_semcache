#!/usr/bin/env bash
#SBATCH --job-name=r5_i5r2b_arithmetic
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodelist=moana-y3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_i5r2b_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5_i5r2b_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?manual allocation required}" "${I5R2B_EXPERIMENT:?required}" "${I5R2B_MULTI_PREPARED:?required}"
: "${I5R2B_RUN_ROOT:?fresh attempt parent required}" "${I5R2B_SOURCE_COMMIT:?reviewed commit required}"
[[ "${R5_ENABLE_GPU:-0}" == 1 && "${R5_I5_ENABLE_GPU:-0}" == 1 ]] || exit 2
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration ]] || exit 2
[[ "$(git rev-parse HEAD)" == "$I5R2B_SOURCE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || exit 2
if [[ -n "${I5_CONDA_ROOT:-}" ]]; then source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"; else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TMPDIR="${SLURM_TMPDIR:-/tmp}"
[[ "${SLURM_ARRAY_TASK_ID:-0}" == 0 ]] || exit 2
mkdir -p "$I5R2B_RUN_ROOT"
python -u scripts/131_run_r5_i5r2b.py --experiment "$I5R2B_EXPERIMENT" --multi-prepared "$I5R2B_MULTI_PREPARED" \
  --source-commit "$I5R2B_SOURCE_COMMIT" --enable-gpu-experiment --runtime-budget-seconds 3000 \
  --output-dir "$I5R2B_RUN_ROOT/block_0_${SLURM_JOB_ID}"
