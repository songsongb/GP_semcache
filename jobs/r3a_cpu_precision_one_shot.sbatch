#!/bin/bash
#SBATCH --job-name=r3a_cpu_precision
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/khuss/r3a_cpu_precision_%j.out
#SBATCH --error=/data/khuss/r3a_cpu_precision_%j.err
set -euo pipefail

# No node pinning or nested Slurm allocation: prepare and all candidates share
# this job's GPU. Existing launchers and all historical runs are untouched.
cd "${R3A_PRECISION_REPO:-/data/khuss/repos/GP_semcache}"
if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo 'Requires one Slurm GPU allocation; preserve CUDA_VISIBLE_DEVICES.' >&2
    exit 2
fi
mkdir -p results/r3/a
r3a_precision_root="results/r3/a/precision_study_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$r3a_precision_root"
r3a_precision_on_exit() {
    local r3a_precision_status=$?
    if [[ "$r3a_precision_status" -ne 0 ]]; then
        printf 'Launcher failed: exit=%s; inspect Slurm stdout/stderr and Python reports.\n' \
            "$r3a_precision_status" > "$r3a_precision_root/launcher_failure.txt"
        if [[ ! -f "$r3a_precision_root/r3a_cpu_precision_results.json" ]]; then
            printf '{"execution_status":"FAILED","classification":"R3A_PARITY_DIAGNOSTIC","PRIMARY_PERFORMANCE_ELIGIBLE":false,"gpu_findings_status":"NOT_RUN","original_151730_evidence_loaded":false,"reason":"launcher environment failed before Python study; inspect Slurm stderr","launcher_exit_code":%s}\n' \
                "$r3a_precision_status" > "$r3a_precision_root/r3a_cpu_precision_results.json"
            printf 'R3A_PARITY_DIAGNOSTIC\nFAILED: launcher environment; GPU findings NOT_RUN. See Slurm stderr.\n' \
                > "$r3a_precision_root/r3a_cpu_precision_summary.txt"
        fi
    fi
}
trap r3a_precision_on_exit EXIT

if [[ -n "${R3A_CONDA_ROOT:-}" ]]; then
    source "$R3A_CONDA_ROOT/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

echo "Run root: $r3a_precision_root; host: $(hostname); job: $SLURM_JOB_ID"
# Script98 prints the UUID of Torch's actual allocated cuda:0, then creates a
# fresh immutable reference through the exact script95 prepare CLI. Each later
# child must match that new UUID; no UUID from a historical job is required.
python -u scripts/98_explore_r3a_cpu_precision.py \
    --phase run --enable-gpu-probes --device cuda:0 --seed 42 \
    --output-dir "$r3a_precision_root"
echo "Diagnostic study completed: $r3a_precision_root/r3a_cpu_precision_results.json"
