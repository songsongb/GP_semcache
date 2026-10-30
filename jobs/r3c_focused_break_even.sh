#!/usr/bin/env bash
#SBATCH --job-name=r3c_focused
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=logs/r3c_focused_%j.out
#SBATCH --error=logs/r3c_focused_%j.err
set -euo pipefail
# Slurm opens its log paths first: create GP_semcache/logs before submission.
cd "${R3CF_REPO:-/data/khuss/repos/GP_semcache}"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo 'One Slurm GPU allocation required.' >&2; exit 2; }
mkdir -p results/r3/c
r3cf_root="results/r3/c/focused_break_even_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$r3cf_root"
trap 'r3cf_status=$?; if [[ "$r3cf_status" -ne 0 ]]; then printf "Launcher failed: exit=%s; retain all traces and inspect logs/r3c_focused_%s.err.\n" "$r3cf_status" "$SLURM_JOB_ID" > "$r3cf_root/launcher_failure.txt"; fi' EXIT
if [[ -n "${R3CF_CONDA_ROOT:-}" ]]; then source "$R3CF_CONDA_ROOT/etc/profile.d/conda.sh";
else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
echo "R3C-F host=$(hostname) job=$SLURM_JOB_ID visible_gpu=$CUDA_VISIBLE_DEVICES output=$r3cf_root"
git branch --show-current
git rev-parse HEAD
git status --short
python scripts/101_run_r3c_break_even_focused.py --help
python -u scripts/101_run_r3c_break_even_focused.py --phase reanalyze \
    --legacy-input "${R3CF_LEGACY_INPUT:-results/r3/c/network_one_shot_151893_20261030T065147Z}" \
    --output-dir "${r3cf_root}_legacy_analysis"
python -u scripts/101_run_r3c_break_even_focused.py --phase run --enable-gpu-experiment \
    --device cuda:0 --seed 42 --resamples 10000 --output-dir "$r3cf_root"
python -u scripts/101_run_r3c_break_even_focused.py --phase analyze --inputs "$r3cf_root" \
    --output-dir "${r3cf_root}_analysis"
echo "Completed focused diagnostic; Native FAIL and UNRESOLVED brackets are valid: $r3cf_root"
