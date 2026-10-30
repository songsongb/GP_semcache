#!/usr/bin/env bash
#SBATCH --job-name=r4b1_multi_entry
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=logs/r4b1_%j.out
#SBATCH --error=logs/r4b1_%j.err
set -euo pipefail
# Opt-in template only. Codex does not submit this job.
cd "${R4B1_REPO:-/data/khuss/repos/GP_semcache}"
[[ "${R4B1_ENABLE_GPU:-0}" == 1 ]] || { echo 'Set R4B1_ENABLE_GPU=1 for this explicit pilot.' >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo 'One allocated RTX3090 required.' >&2; exit 2; }
[[ "$(git branch --show-current)" == exp/raw-semcache-r4b ]] || { echo 'Wrong branch.' >&2; exit 2; }
if [[ -n "${R4B1_CONDA_ROOT:-}" ]]; then source "$R4B1_CONDA_ROOT/etc/profile.d/conda.sh";
else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p results/r4/b1
r4b1_root="results/r4/b1/opt_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$r4b1_root"
trap 'r4b1_status=$?; if [[ "$r4b1_status" -ne 0 ]]; then printf "FAILED exit=%s; retain partial evidence.\n" "$r4b1_status" > "$r4b1_root/launcher_failure.txt"; fi' EXIT
python scripts/103_run_r4b1_multi_entry.py --phase dry-run --output-dir "$r4b1_root/preflight"
python -c 'import importlib.util; names=("torch","numpy","transformers","peft","sacrebleu"); assert all(importlib.util.find_spec(n) for n in names), "Missing GPU dependencies"; import torch; assert torch.cuda.device_count()==1 and "3090" in torch.cuda.get_device_name(0), "One RTX3090 required"'
python -u scripts/95_run_r3a_distributed_no_reuse.py --phase prepare --model facebook/opt-2.7b --dtype float16 \
  --prompt-lengths 32 --device cuda:0 --adapter user_a --seed 42 --output-dir "$r4b1_root/base_prepare"
python -u scripts/103_run_r4b1_multi_entry.py --phase prepare --enable-gpu-experiment \
  --base-prepared "$r4b1_root/base_prepare" --output-dir "$r4b1_root/prepare"
python -u scripts/103_run_r4b1_multi_entry.py --phase run --enable-gpu-experiment \
  --prepared "$r4b1_root/prepare/native" --capacity-bytes "${R4B1_CAPACITY_BYTES:-67108864}" --output-dir "$r4b1_root/run"
python scripts/103_run_r4b1_multi_entry.py --phase analyze --inputs "$r4b1_root/run" --output-dir "$r4b1_root/analysis"
echo "R4-B1 diagnostic complete. Native strict parity remains independently gated: $r4b1_root"
