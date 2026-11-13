#!/bin/bash
# Manual template only. No submission has been performed.
#SBATCH --job-name=d5-light-REVIEW-BLOCKED
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:30:00
set -euo pipefail
# Deliberate guard until native ES/UD worker, dependencies, real plans and
# clean-commit CPU approval are implemented and reviewed. Never fall back to Strict.
echo 'BLOCKED: native Light worker not implemented/validated. See docs/D5_L1_L5_GPU_RUNBOOK.md.' >&2
exit 2
