#!/bin/bash
#SBATCH --job-name=r1b1_impact
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=23:50:00
#SBATCH --array=0-1%1
#SBATCH --output=/data/khuss/r1b1_%A_%a.out
#SBATCH --error=/data/khuss/r1b1_%A_%a.err
set -euo pipefail
cd /data/khuss/repos/GP_semcache
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false

lengths=(32 64)
r1b1_length="${lengths[${SLURM_ARRAY_TASK_ID:-0}]}"
r1b1_tag="${R1B1_RUN_TAG:-opt27b_impact_v1}"
if [[ ! "$r1b1_tag" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo 'R1B1_RUN_TAG must be a single alphanumeric/underscore/hyphen component' >&2
    exit 2
fi
resume_args=()
if [[ "${R1B1_RESUME:-0}" == 1 ]]; then resume_args=(--resume); fi
python3 -u scripts/70_run_r1b1_impact.py \
    --model facebook/opt-2.7b --dtype float16 --device cuda:0 \
    --prompt-lengths "$r1b1_length" \
    --cache-devices "${R1B1_STORAGE:-cpu,cuda}" \
    --instrumentation "${R1B1_INSTRUMENTATION:-both}" \
    --impact-executions "${R1B1_EXECUTIONS:-legacy_blockwise,token_precompute}" \
    --warmup-runs 3 --measured-runs 10 \
    --output-dir "results/r1b1/${r1b1_tag}/tokens_${r1b1_length}" \
    "${resume_args[@]}"
