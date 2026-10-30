#!/usr/bin/env bash
#SBATCH --job-name=r4b2_prefix_kv
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=logs/r4b2_%j.out
#SBATCH --error=logs/r4b2_%j.err
set -euo pipefail
# Opt-in template; no submission is performed by the implementation task.
cd "${R4B2_REPO:-/data/khuss/repos/GP_semcache}"
[[ "${R4B2_ENABLE_GPU:-0}" == 1 ]] || { echo 'Explicit R4B2_ENABLE_GPU=1 required.' >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo 'One allocated RTX3090 required.' >&2; exit 2; }
[[ "$(git branch --show-current)" == exp/raw-semcache-r4b2 ]] || { echo 'Wrong work branch.' >&2; exit 2; }
if [[ -n "${R4B2_CONDA_ROOT:-}" ]]; then source "$R4B2_CONDA_ROOT/etc/profile.d/conda.sh";
else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p results/r4/b2
r4b2_root="results/r4/b2/opt_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$r4b2_root"
trap 'r4b2_status=$?; if [[ "$r4b2_status" -ne 0 ]]; then printf "FAILED exit=%s; preserve all partial evidence.\n" "$r4b2_status" > "$r4b2_root/launcher_failure.txt"; fi' EXIT
python scripts/104_run_r4b2_shared_prefix.py --phase dry-run --output-dir "$r4b2_root/preflight"
python -c 'import importlib.util; names=("torch","numpy","transformers","peft","sacrebleu"); assert all(importlib.util.find_spec(n) for n in names), "Missing dependencies"; import torch; assert torch.cuda.device_count()==1 and "3090" in torch.cuda.get_device_name(0), "One RTX3090 required"'
python -u scripts/95_run_r3a_distributed_no_reuse.py --phase prepare --model facebook/opt-2.7b --dtype float16 \
  --prompt-lengths 32 --device cuda:0 --adapter user_a --seed 42 --output-dir "$r4b2_root/base_prepare"
python -u scripts/104_run_r4b2_shared_prefix.py --phase prepare --enable-gpu-experiment \
  --base-prepared "$r4b2_root/base_prepare" --target-length 128 --max-new-tokens 8 --prefix-lengths 32 64 96 --output-dir "$r4b2_root/prepare"
python -u scripts/104_run_r4b2_shared_prefix.py --phase run --enable-gpu-experiment \
  --prepared "$r4b2_root/prepare/prepared" --capacity-bytes 134217728 --output-dir "$r4b2_root/run"
python scripts/104_run_r4b2_shared_prefix.py --phase analyze --inputs "$r4b2_root/run" --output-dir "$r4b2_root/analysis"
echo "B2 complete; inspect independent strict fidelity and numerical deviations before any performance interpretation: $r4b2_root"
