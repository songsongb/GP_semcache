#!/bin/bash
#SBATCH --job-name=r3b_raw_qkv
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=logs/r3b_raw_qkv_%j.out
#SBATCH --error=logs/r3b_raw_qkv_%j.err
set -euo pipefail

# Create GP_semcache/logs before submission: Slurm opens log files before Bash.
cd "${R3B_REPO:-/data/khuss/repos/GP_semcache}"
if [[ -n "${R3B_CONDA_ROOT:-}" ]]; then
    source "$R3B_CONDA_ROOT/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo 'Submit within one Slurm GPU allocation; preserve CUDA_VISIBLE_DEVICES.' >&2
    exit 2
fi
mkdir -p results/r3/b
r3b_root="results/r3/b/raw_qkv_one_shot_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$r3b_root"
trap 'r3b_exit=$?; if [[ "$r3b_exit" -ne 0 ]]; then printf "Launcher failed: exit=%s; see logs/r3b_raw_qkv_%s.err and preserved Python evidence.\n" "$r3b_exit" "$SLURM_JOB_ID" > "$r3b_root/launcher_failure.txt"; fi' EXIT
echo "R3-B allocation host=$(hostname) job=$SLURM_JOB_ID output=$r3b_root"
python -u scripts/99_run_r3b_raw_qkv_one_shot.py --phase run --enable-gpu-experiment \
    --device cuda:0 --warmup-runs 1 --measured-runs 5 --cache-capacity-mib 64 \
    --output-dir "$r3b_root"
echo "R3-B diagnostic complete: $r3b_root/r3b_one_shot_results.json"
