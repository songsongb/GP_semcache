#!/usr/bin/env bash
#SBATCH --job-name=r5_d5_natural
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:30:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_d5_natural_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5_d5_natural_%j.err
set -Eeuo pipefail
: "${SLURM_JOB_ID:?manual allocation required}" "${D5N_EXPERIMENT:?sealed one-block plan required}"
: "${D5N_CPU_AUDIT:?native CPU audit required}" "${D5N_SOURCE_COMMIT:?reviewed source required}"
: "${D5N_RUN_ROOT:?fresh attempts parent required}" "${D5N_NODE:?select moana-y3 or moana-y4}" "${I5_CONDA_ROOT:?Conda root required}"
[[ "$D5N_NODE" == moana-y3 || "$D5N_NODE" == moana-y4 ]] || exit 2
[[ "${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-}}" == "$D5N_NODE" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 2
[[ "${R5_ENABLE_GPU:-0}" == 1 && "${R5_I5_ENABLE_GPU:-0}" == 1 ]] || exit 2
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration && "$(git rev-parse HEAD)" == "$D5N_SOURCE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || exit 2
[[ -d logs && -d "$D5N_RUN_ROOT" && -d "$D5N_EXPERIMENT" && -d "$D5N_CPU_AUDIT" && ! -e "$D5N_RUN_ROOT/run_${SLURM_JOB_ID}" ]] || exit 2
source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=42
CUDA_VISIBLE_DEVICES='' python scripts/153_r5_subsequence_natural.py inspect --experiment "$D5N_EXPERIMENT" --cpu-audit "$D5N_CPU_AUDIT"
printf 'D5 NATURAL plan=%s CPU-audit=%s run=%s\n' "$D5N_EXPERIMENT" "$D5N_CPU_AUDIT" "$D5N_RUN_ROOT/run_${SLURM_JOB_ID}"
timeout --signal=TERM --kill-after=60s 241m python -u scripts/153_r5_subsequence_natural.py run \
 --experiment "$D5N_EXPERIMENT" --cpu-audit "$D5N_CPU_AUDIT" --source-commit "$D5N_SOURCE_COMMIT" \
 --output-dir "$D5N_RUN_ROOT/run_${SLURM_JOB_ID}" --enable-gpu-experiment
