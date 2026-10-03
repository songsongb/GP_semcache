#!/usr/bin/env bash
#SBATCH --job-name=c6b1-snips
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/c6b1-train-%j.log
set -euo pipefail

# Use the site's existing Python environment; never install or download here.
# Wall time is a scheduling limit, not a measured runtime prediction.
selection_path=${1:?Usage: sbatch scripts/50_train_cachegen_c6b_snips.sbatch /absolute/selection.json /absolute/new/output-root}
output_path=${2:?Pass a new empty output root}
export TORCH_EXTENSIONS_DIR=/data/khuss/.cache/torch_extensions/py311_cu118
export TRITON_CACHE_DIR=/data/khuss/.cache/triton
export CUDA_CACHE_PATH=/data/khuss/.cache/cuda
export TORCH_HOME=/data/khuss/.cache/torch
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export XDG_CACHE_HOME=/data/khuss/.cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python3 -u scripts/50_train_cachegen_c6b_snips.py \
  --snips results/workloads/m9b_snips_semantic.jsonl \
  --selection "$selection_path" --output-root "$output_path" \
  --per-intent-per-user 200 --epochs 3 --learning-rate 0.0001 \
  --gradient-accumulation 8 --max-sequence-length 256 \
  --gradient-checkpointing --seed 42 --device cuda:0
