#!/usr/bin/env bash
#SBATCH --job-name=r5_i5r2c3_attention
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_i5r2c3_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5_i5r2c3_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?manual allocation required}" "${C3_NODE:?select moana-y3 or moana-y4}"
: "${C3_ENABLE_GPU:?explicit 1 required}" "${C3_EXPERIMENT:?sealed replay plan required}"
: "${C3_SOURCE_COMMIT:?reviewed source commit required}" "${C3_RUN_ROOT:?fresh output parent required}"
: "${I5_CONDA_ROOT:?Conda installation path required}"
[[ "$C3_ENABLE_GPU" == 1 && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "C3_ENABLE_GPU=1 and a non-array allocation required" >&2; exit 2; }
[[ "$C3_NODE" == moana-y3 || "$C3_NODE" == moana-y4 ]] || { echo "C3_NODE must be moana-y3 or moana-y4" >&2; exit 2; }
[[ "${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-}}" == "$C3_NODE" ]] || { echo 'Allocated node mismatch' >&2; exit 2; }
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration && "$(git rev-parse HEAD)" == "$C3_SOURCE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Pinned source/branch/clean-tree mismatch' >&2; exit 2; }
source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# Set CUBLAS_WORKSPACE_CONFIG before submission to match the historical receipt.
mkdir -p "$C3_RUN_ROOT"
python -u scripts/139_r5_i5r2c3.py run --experiment "$C3_EXPERIMENT" \
 --source-commit "$C3_SOURCE_COMMIT" --output-dir "$C3_RUN_ROOT/run_${SLURM_JOB_ID}"
