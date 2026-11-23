#!/usr/bin/env bash
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_RESTART_COUNT:-0}" == 0 && $# == 3 ]]
entry="$(pwd -P)/scripts/165_r5_d5_l6b.py"
[[ -f "$entry" && -x "$1" ]]
"$1" "$entry" _registered --trial "$2" --condition "$3"
exec srun --nodes=1 --ntasks=1 --cpus-per-task=4 --cpu-bind=cores "$1" "$entry" _job --trial "$2" --condition "$3"
