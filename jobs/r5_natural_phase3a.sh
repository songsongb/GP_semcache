#!/usr/bin/env bash
#SBATCH --job-name=r5_d5_phase3a
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:30:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_d5_phase3a_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5_d5_phase3a_%j.err
set -Eeuo pipefail
: "${SLURM_JOB_ID:?manual allocation required}" "${D5NP_EXPERIMENT:?sealed one-dataset Phase3A plan required}"
: "${D5NP_CPU_AUDIT:?native zero-skip CPU approval required}" "${D5NP_SOURCE_COMMIT:?reviewed commit required}"
: "${D5NP_RUN_ROOT:?fresh attempts parent required}" "${D5N_NODE:?select moana-y3 or moana-y4}" "${I5_CONDA_ROOT:?existing Conda root required}"
[[ "$D5N_NODE" == moana-y3 || "$D5N_NODE" == moana-y4 ]] || exit 2
[[ "${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-}}" == "$D5N_NODE" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 2
[[ "${R5_ENABLE_GPU:-0}" == 1 && "${R5_I5_ENABLE_GPU:-0}" == 1 ]] || exit 2
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration && "$(git rev-parse HEAD)" == "$D5NP_SOURCE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || exit 2
[[ -d logs && -d "$D5NP_RUN_ROOT" && ! -e "$D5NP_RUN_ROOT/run_${SLURM_JOB_ID}" ]] || exit 2
source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=42
CUDA_VISIBLE_DEVICES='' python scripts/158_r5_natural_phase3a.py inspect --experiment "$D5NP_EXPERIMENT" --cpu-audit "$D5NP_CPU_AUDIT"
timeout --signal=TERM --kill-after=60s 241m python -u scripts/158_r5_natural_phase3a.py run \
 --experiment "$D5NP_EXPERIMENT" --cpu-audit "$D5NP_CPU_AUDIT" --source-commit "$D5NP_SOURCE_COMMIT" \
 --output-dir "$D5NP_RUN_ROOT/run_${SLURM_JOB_ID}" --enable-gpu-experiment --enable-approximate-experiment
