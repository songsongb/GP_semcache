#!/usr/bin/env bash
#SBATCH --job-name=r5_i5r1_controlled
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodelist=moana-y4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_i5r1_%A_%a.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5_i5r1_%A_%a.err
set -euo pipefail
: "${SLURM_JOB_ID:?manual GPU allocation required}"
: "${I5R1_EXPERIMENT:?absolute sealed experiment path required}"
: "${I5R1_MULTI_PREPARED:?verified three-adapter preparation required}"
: "${I5R1_RUN_ROOT:?fresh experiment attempt parent required}"
: "${I5R1_SOURCE_COMMIT:?pinned reviewed commit required}"
[[ "${R5_I5_ENABLE_GPU:-0}" == 1 && "${R5_ENABLE_GPU:-0}" == 1 ]] || exit 2
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration ]] || exit 2
[[ "$(git rev-parse HEAD)" == "$I5R1_SOURCE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || exit 2
if [[ -n "${I5_CONDA_ROOT:-}" ]]; then source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"; else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$I5R1_RUN_ROOT"
python -u scripts/127_run_r5_i5r1.py --experiment "$I5R1_EXPERIMENT" \
  --multi-prepared "$I5R1_MULTI_PREPARED" --block-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --source-commit "$I5R1_SOURCE_COMMIT" --enable-gpu-experiment \
  --runtime-budget-seconds "${I5R1_RUNTIME_SECONDS:-9000}" \
  --output-dir "$I5R1_RUN_ROOT/block_${SLURM_ARRAY_TASK_ID:-0}_${SLURM_JOB_ID}"
