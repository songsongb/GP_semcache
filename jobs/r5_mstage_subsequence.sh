#!/usr/bin/env bash
# Deliberately NOT a submission-ready GPU job: physical integration is blocked.
# Run this CPU readiness gate before allocating any GPU. No sbatch or inference.
set -Eeuo pipefail
: "${MS_EXPERIMENT:?sealed integration plan required}"
: "${MS_SOURCE_COMMIT:?reviewed source commit required}"
python scripts/150_r5_mstage.py gpu-preflight \
 --experiment "$MS_EXPERIMENT" --source-commit "$MS_SOURCE_COMMIT"
echo 'Refusing launch: no audited distributed subsequence executor exists' >&2
exit 2
