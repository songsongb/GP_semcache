#!/usr/bin/env bash
#SBATCH --job-name=r3c_network
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=logs/r3c_network_%j.out
#SBATCH --error=logs/r3c_network_%j.err
set -euo pipefail
cd "${R3C_REPO:-/data/khuss/repos/GP_semcache}"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo 'One Slurm GPU allocation required.' >&2; exit 2; }
mkdir -p results/r3/c
r3c_root="results/r3/c/network_one_shot_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$r3c_root"
trap 'r3c_status=$?; if [[ "$r3c_status" -ne 0 ]]; then printf "Launcher failed: exit=%s; inspect Slurm stderr and preserved Python artifacts.\n" "$r3c_status" > "$r3c_root/launcher_failure.txt"; fi' EXIT
if [[ -n "${R3C_CONDA_ROOT:-}" ]]; then source "$R3C_CONDA_ROOT/etc/profile.d/conda.sh";
else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
echo "R3-C host=$(hostname) allocation=$SLURM_JOB_ID visible_gpu=$CUDA_VISIBLE_DEVICES output=$r3c_root"
git branch --show-current
git rev-parse HEAD
git status --short
python -u scripts/100_run_r3c_network_one_shot.py --phase run --enable-gpu-experiment \
    --device cuda:0 --rtt-ms 0,10 --bandwidth-mbps unlimited,200 \
    --hit-fractions 0,0.125,0.25,0.5,1 --requests-per-arm 8 --warmup-runs 1 --seed 42 \
    --output-dir "$r3c_root"
python -u scripts/100_run_r3c_network_one_shot.py --phase analyze --inputs "$r3c_root" \
    --output-dir "${r3c_root}_analysis"
echo "Completed diagnostic (Native FAIL/no crossover may be valid): $r3c_root"
