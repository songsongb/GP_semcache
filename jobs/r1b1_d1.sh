#!/bin/bash
#SBATCH --job-name=r1b1_d1
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=23:50:00
#SBATCH --array=0-1%1
#SBATCH --output=/data/khuss/r1b1_d1_%A_%a.out
#SBATCH --error=/data/khuss/r1b1_d1_%A_%a.err
set -euo pipefail
cd /data/khuss/repos/GP_semcache
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
: "${D1_REFERENCE_ROOT:?Set the existing R1-B1 run root containing tokens_32 and tokens_64}"
d1_lengths=(32 64)
d1_length="${d1_lengths[${SLURM_ARRAY_TASK_ID:-0}]}"
d1_tag="${D1_RUN_TAG:-diagnosis_v1}"
if [[ ! "$d1_tag" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo 'D1_RUN_TAG must be one alphanumeric/underscore/hyphen component' >&2
    exit 2
fi
python3 -u scripts/71_diagnose_r1b1_correctness.py \
    --raw "$D1_REFERENCE_ROOT/tokens_$d1_length/raw.jsonl" \
    --lengths "$d1_length" --device cuda:0 \
    --repeat "${D1_REPEAT:-0}" \
    --impact-execution "${D1_IMPACT_EXECUTION:-token_precompute}" \
    --storage "${D1_STORAGE:-RAW_CPU}" --instrumentation "${D1_INSTRUMENTATION:-minimal}" \
    --output-dir "results/r1b1_d1/$d1_tag/tokens_$d1_length"
