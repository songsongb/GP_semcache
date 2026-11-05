#!/usr/bin/env bash
#SBATCH --job-name=r5_i5r2c4_scores
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_i5r2c4_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5_i5r2c4_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?manual allocation required}" "${C4_ENABLE_GPU:?explicit 1 required}"
: "${C4_NODE:?select moana-y3 or moana-y4}" "${C4_EXPERIMENT:?sealed plan required}"
: "${C4_SOURCE_COMMIT:?reviewed source required}" "${C4_RUN_ROOT:?fresh attempt parent required}"
: "${I5_CONDA_ROOT:?Conda installation path required}"
[[ "$C4_ENABLE_GPU" == 1 && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo 'One opted-in non-array job required' >&2; exit 2; }
[[ "$C4_NODE" == moana-y3 || "$C4_NODE" == moana-y4 ]] || { echo 'Unsupported C4_NODE' >&2; exit 2; }
[[ "${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-}}" == "$C4_NODE" ]] || { echo 'Allocated node mismatch' >&2; exit 2; }
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration && "$(git rev-parse HEAD)" == "$C4_SOURCE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Pinned source/branch/clean-tree mismatch' >&2; exit 2; }
source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$C4_RUN_ROOT"
python -u scripts/141_r5_i5r2c4.py run --experiment "$C4_EXPERIMENT" \
 --source-commit "$C4_SOURCE_COMMIT" --output-dir "$C4_RUN_ROOT/run_${SLURM_JOB_ID}"
