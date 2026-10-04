#!/bin/bash
#SBATCH --job-name=c6b3_full
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=/data/khuss/c6b3_full_%j.out
#SBATCH --error=/data/khuss/c6b3_full_%j.err
set -euo pipefail
cd /data/khuss/repos/GP_semcache
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export XDG_CACHE_HOME=/data/khuss/.cache TORCH_HOME=/data/khuss/.cache/torch
export TORCH_EXTENSIONS_DIR=/data/khuss/.cache/torch_extensions/py311_cu118
export TRITON_CACHE_DIR=/data/khuss/.cache/triton CUDA_CACHE_PATH=/data/khuss/.cache/cuda
export TMPDIR=/data/khuss/.cache/c6b3_full_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/54_train_cachegen_c6b3_multiwoz.py --profile full   --output-root results/cachegen/c6b3/b1_full
python scripts/55_eval_cachegen_c6b3_multiwoz.py   --adapter-root results/cachegen/c6b3/b1_full   --output-root results/cachegen/c6b3/b1_full_capability64
