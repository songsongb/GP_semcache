#!/usr/bin/env bash
#SBATCH --job-name=r5_i5r2c2_linear
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_i5r2c2_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5_i5r2c2_%j.err
# Node is deliberately supplied by the operator with --nodelist="$C2_NODE".
set -euo pipefail
: "${SLURM_JOB_ID:?manual Slurm allocation required}" "${C2_ENABLE_GPU:?explicit value 1 required}"
: "${C2_NODE:?operator-selected permitted GPU node required}" "${C2_EXPERIMENT:?sealed plan required}"
: "${C2_SOURCE_COMMIT:?reviewed commit required}" "${C2_RUN_ROOT:?fresh attempt parent required}"
: "${C2_MULTI_PREPARED:?three-adapter preparation required}"
[[ "$C2_ENABLE_GPU" == 1 && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo 'One opted-in job only; arrays are forbidden' >&2; exit 2; }
[[ "$C2_NODE" == moana-y3 || "$C2_NODE" == moana-y4 ]] || { echo 'Choose a permitted moana-y3 or moana-y4 allocation' >&2; exit 2; }
[[ "${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-}}" == "$C2_NODE" ]] || { echo 'Allocated node differs from C2_NODE' >&2; exit 2; }
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration && "$(git rev-parse HEAD)" == "$C2_SOURCE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Pinned source/branch/clean-tree check failed' >&2; exit 2; }
if [[ -n "${I5_CONDA_ROOT:-}" ]]; then source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"; else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export R5_I5_ENABLE_GPU=1 R5_ENABLE_GPU=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$C2_RUN_ROOT"
python -u scripts/138_r5_i5r2c2.py run --experiment "$C2_EXPERIMENT" \
  --multi-prepared "$C2_MULTI_PREPARED" --source-commit "$C2_SOURCE_COMMIT" \
  --enable-gpu-experiment --output-dir "$C2_RUN_ROOT/run_${SLURM_JOB_ID}"
