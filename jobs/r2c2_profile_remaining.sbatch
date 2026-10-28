#!/bin/bash
#SBATCH --job-name=r2c2_profile
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=23:50:00
#SBATCH --output=/data/khuss/r2c2_%j.out
#SBATCH --error=/data/khuss/r2c2_%j.err
set -euo pipefail

# One allocation, three sequential Python runs, no array/srun rescheduling.
cd "${R2C2_REPO:-/data/khuss/repos/GP_semcache}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semcache
export HF_HOME="${HF_HOME:-/data/khuss/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false

if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo 'Run with sbatch in a GPU allocation; CUDA_VISIBLE_DEVICES is preserved.' >&2
    exit 2
fi
r2c2_tag="${R2C2_RUN_TAG:-remaining}"
if [[ ! "$r2c2_tag" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo 'R2C2_RUN_TAG must contain only letters, digits, underscore or hyphen.' >&2
    exit 2
fi
mkdir -p results/r2/c2
r2c2_root="results/r2/c2/${r2c2_tag}_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$r2c2_root"
r2c2_launcher="$(realpath "${BASH_SOURCE[0]}")"
python -u scripts/91_validate_r2c2_results.py identity \
    --device cuda:0 --launcher "$r2c2_launcher" --output "$r2c2_root/launch_identity.json"

r2c2_common=(--model facebook/opt-2.7b --dtype float16 --device cuda:0
    --cache-device cuda --capacity-mib 512 --seed 42)

r2c2_run() {
    local r2c2_output="$1"
    shift
    python scripts/91_validate_r2c2_results.py check-code --identity "$r2c2_root/launch_identity.json"
    local r2c2_exit=0
    python -u scripts/90_run_r2c1_long_context_qkv.py "${r2c2_common[@]}" \
        --output-dir "$r2c2_output" "$@" || r2c2_exit=$?
    # Runs even after a Python exception. COMPLETE_WITH_FAILURES prints original
    # case status, reason, traceback and both memory budgets before exiting.
    python -u scripts/91_validate_r2c2_results.py validate \
        --identity "$r2c2_root/launch_identity.json" --directory "$r2c2_output" || return 1
    if [[ "$r2c2_exit" -ne 0 ]]; then
        echo "Runner failed with exit=$r2c2_exit; artifacts: $r2c2_output" >&2
        return "$r2c2_exit"
    fi
}

# Fresh code-compatible Pilot; do not edit/reuse historical approval JSON.
r2c2_run "$r2c2_root/pilot" --phase pilot --prompt-lengths 512,1024 \
    --coverages 0,.5,max --warmup-runs 1 --measured-runs 3 --independent-run-id c2_pilot

# Only remaining L1024 traces. L512 profiling is deliberately not repeated.
r2c2_run "$r2c2_root/profile_1024_0_max" --phase profile --prompt-lengths 1024 \
    --coverages 0,max --pilot-inputs "$r2c2_root/pilot" --profile-memory-policy pilot_guarded \
    --profile-memory-allowance-gib 1 --profile-memory-safety-gib 2 \
    --max-trace-mib "${R2C2_MAX_TRACE_MIB:-64}" --independent-run-id c2_profile_0_max
r2c2_run "$r2c2_root/profile_1024_half" --phase profile --prompt-lengths 1024 \
    --coverages .5 --pilot-inputs "$r2c2_root/pilot" --profile-memory-policy pilot_guarded \
    --profile-memory-allowance-gib 1 --profile-memory-safety-gib 2 \
    --max-trace-mib "${R2C2_MAX_TRACE_MIB:-64}" --independent-run-id c2_profile_half
echo "R2-C2 remaining profiles validated: $r2c2_root"
