#!/usr/bin/env bash
#SBATCH --job-name=r5_i5r2c3b0_capture
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_i5r2c3b0_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5_i5r2c3b0_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?manual allocation required}" "${B0_NODE:?select moana-y3 or moana-y4}"
: "${B0_ENABLE_GPU:?explicit 1 required}" "${B0_EXPERIMENT:?sealed B0 plan required}"
: "${B0_SOURCE_COMMIT:?reviewed source commit required}" "${B0_RUN_ROOT:?fresh attempt parent required}"
: "${I5_CONDA_ROOT:?Conda installation path required}"
[[ "$B0_ENABLE_GPU" == 1 && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo 'One opted-in non-array job required' >&2; exit 2; }
[[ "$B0_NODE" == moana-y3 || "$B0_NODE" == moana-y4 ]] || { echo 'Unsupported B0_NODE' >&2; exit 2; }
[[ "${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-}}" == "$B0_NODE" ]] || { echo 'Allocated node mismatch' >&2; exit 2; }
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration && "$(git rev-parse HEAD)" == "$B0_SOURCE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Pinned source/branch/clean-tree mismatch' >&2; exit 2; }
source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false R5_I5_ENABLE_GPU=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# Operator must match historical CUBLAS_WORKSPACE_CONFIG before Python starts.
mkdir -p "$B0_RUN_ROOT"
python -u scripts/140_r5_i5r2c3b0.py run --experiment "$B0_EXPERIMENT" \
 --source-commit "$B0_SOURCE_COMMIT" --output-dir "$B0_RUN_ROOT/run_${SLURM_JOB_ID}"
