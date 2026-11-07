#!/usr/bin/env bash
#SBATCH --job-name=r5_i5r4
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_i5r4_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5_i5r4_%j.err
# The user supplies --nodelist="$I4_NODE" at sbatch submission.
# This job runs one GPU campaign and automatically audits/verifies/reports it.
# Requires the I5R-4 CLI implemented to the contract below.
set -Eeuo pipefail
stage=START
trap 'rc=$?; echo "I5R4_ABORTED stage=${stage} exit=${rc} job=${SLURM_JOB_ID:-unknown}" >&2; exit "$rc"' ERR

: "${SLURM_JOB_ID:?manual Slurm allocation required}"
: "${I4_EXPERIMENT:?sealed I5R4 experiment required}"
: "${I4_CPU_AUDIT:?sealed CPU preflight validation required}"
: "${I4_SOURCE_COMMIT:?reviewed committed source required}"
: "${I4_RUN_ROOT:?fresh run parent required}"
: "${I4_NODE:?operator-selected moana-y3 or moana-y4 required}"
: "${I5_CONDA_ROOT:?Conda base path required}"

[[ "$I4_NODE" == moana-y3 || "$I4_NODE" == moana-y4 ]] || { echo 'Unsupported GPU node' >&2; exit 2; }
[[ "${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-}}" == "$I4_NODE" ]] || { echo 'Allocated node differs from I4_NODE' >&2; exit 2; }
[[ "${I4_ENABLE_GPU:-0}" == 1 && "${R5_I5_ENABLE_GPU:-0}" == 1 && "${R5_ENABLE_GPU:-0}" == 1 ]] || { echo 'Explicit GPU opt-in required' >&2; exit 2; }
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo 'Job arrays forbidden for this campaign' >&2; exit 2; }
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration && "$(git rev-parse HEAD)" == "$I4_SOURCE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Pinned source/branch/clean-tree check failed' >&2; exit 2; }
[[ -d "$I4_EXPERIMENT" ]] || { echo 'I4_EXPERIMENT missing' >&2; exit 2; }
[[ -d "$I4_CPU_AUDIT" ]] || { echo 'I4_CPU_AUDIT missing' >&2; exit 2; }
[[ -d "$I4_RUN_ROOT" ]] || { echo 'I4_RUN_ROOT parent must exist before submission' >&2; exit 2; }

source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate semcache
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="${SLURM_TMPDIR:-/tmp}"

I4_RUN="$I4_RUN_ROOT/run_${SLURM_JOB_ID}"
I4_AUDIT="${I4_RUN}_audit"
I4_VERIFY="${I4_RUN}_verification"
I4_REPORT="${I4_RUN}_report"
for p in "$I4_RUN" "$I4_AUDIT" "$I4_VERIFY" "$I4_REPORT"; do
  [[ ! -e "$p" && ! -L "$p" ]] || { echo "Refusing non-fresh output: $p" >&2; exit 2; }
done

stage=PREFLIGHT
printf '\n[I5R4 %s] %s\n' "$stage" "$(date -u +%FT%TZ)"
python -u scripts/148_r5_i5r4.py inspect \
  --experiment "$I4_EXPERIMENT" \
  --cpu-audit "$I4_CPU_AUDIT" \
  --source-commit "$I4_SOURCE_COMMIT" \
  --require-cpu-validation

stage=GPU_CAMPAIGN
printf '\n[I5R4 %s] %s\n' "$stage" "$(date -u +%FT%TZ)"
timeout --signal=TERM --kill-after=120s 19h \
  python -u scripts/148_r5_i5r4.py run \
    --experiment "$I4_EXPERIMENT" \
    --source-commit "$I4_SOURCE_COMMIT" \
    --output-dir "$I4_RUN" \
    --runtime-budget-seconds 68400 \
    --enable-gpu-experiment

stage=INDEPENDENT_ANALYSIS
printf '\n[I5R4 %s] %s\n' "$stage" "$(date -u +%FT%TZ)"
timeout --signal=TERM --kill-after=120s 3h \
  python -u scripts/148_r5_i5r4.py analyze \
    --experiment "$I4_EXPERIMENT" \
    --run "$I4_RUN" \
    --cpu-audit "$I4_CPU_AUDIT" \
    --output-dir "$I4_AUDIT"

stage=INDEPENDENT_VERIFY
printf '\n[I5R4 %s] %s\n' "$stage" "$(date -u +%FT%TZ)"
timeout --signal=TERM --kill-after=120s 1h \
  python -u scripts/148_r5_i5r4.py verify \
    --experiment "$I4_EXPERIMENT" \
    --run "$I4_RUN" \
    --audit "$I4_AUDIT" \
    --cpu-audit "$I4_CPU_AUDIT" \
    --output-dir "$I4_VERIFY"

stage=FINAL_REPORT
printf '\n[I5R4 %s] %s\n' "$stage" "$(date -u +%FT%TZ)"
timeout --signal=TERM --kill-after=120s 20m \
  python -u scripts/148_r5_i5r4.py report \
    --experiment "$I4_EXPERIMENT" \
    --run "$I4_RUN" \
    --audit "$I4_AUDIT" \
    --verification "$I4_VERIFY" \
    --output-dir "$I4_REPORT"

stage=COMPLETE
printf '\nI5R4_CAMPAIGN_COMPLETE run=%s audit=%s verification=%s report=%s\n' \
  "$I4_RUN" "$I4_AUDIT" "$I4_VERIFY" "$I4_REPORT"

