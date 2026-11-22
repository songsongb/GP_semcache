#!/usr/bin/env bash
# Manual CLI; GPU allocation only through an explicit `submit --cells ...`.
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:30:00
set -euo pipefail
D5_SWEEP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
D5_SWEEP_BIN="${D5_SWEEP_PYTHON:-python3}"
cd -- "$D5_SWEEP_ROOT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ "${1:-}" == _job ]]; then
  [[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 2
  # One bound step; all three native condition processes inherit this affinity.
  if srun --nodes=1 --ntasks=1 --cpus-per-task=4 --cpu-bind=cores \
    "$D5_SWEEP_BIN" "$D5_SWEEP_ROOT/scripts/163_r5_d5_sweep_v5.py" "$@"; then
    exit 0
  else
    D5_SWEEP_RC=$?
    printf '{"status":"PARTIAL_BLOCKED","stage":"slurm_step_or_python","exit_code":%s}\n' "$D5_SWEEP_RC" >&2
    exit "$D5_SWEEP_RC"
  fi
fi
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES=''
exec "$D5_SWEEP_BIN" "$D5_SWEEP_ROOT/scripts/163_r5_d5_sweep_v5.py" "$@"
