#!/usr/bin/env bash
#SBATCH --job-name=r5_subseq_gen
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:30:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_subseq_gen_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5_subseq_gen_%j.err
set -Eeuo pipefail
: "${SLURM_JOB_ID:?manual allocation required}" "${SSG_EXPERIMENT:?sealed pilot plan required}"
: "${SSG_CPU_AUDIT:?native CPU approval required}" "${SSG_SOURCE_COMMIT:?reviewed source commit required}"
: "${SSG_RUN_ROOT:?fresh output parent required}" "${SSG_NODE:?moana-y3 or moana-y4 required}"
: "${I5_CONDA_ROOT:?Conda root required}"
[[ "$SSG_NODE" == moana-y3 || "$SSG_NODE" == moana-y4 ]] || exit 2
[[ "${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-}}" == "$SSG_NODE" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 2
[[ "${R5_I5_ENABLE_GPU:-0}" == 1 && "${R5_ENABLE_GPU:-0}" == 1 ]] || exit 2
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration && "$(git rev-parse HEAD)" == "$SSG_SOURCE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || exit 2
[[ -d logs && -d "$SSG_RUN_ROOT" && -d "$SSG_EXPERIMENT" && -d "$SSG_CPU_AUDIT" ]] || exit 2
source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=42
timeout --signal=TERM --kill-after=30s 3h python -u scripts/151_r5_subsequence_generation.py run \
 --experiment "$SSG_EXPERIMENT" --cpu-audit "$SSG_CPU_AUDIT" --source-commit "$SSG_SOURCE_COMMIT" \
 --output-dir "$SSG_RUN_ROOT/run_${SLURM_JOB_ID}" --enable-gpu-experiment
