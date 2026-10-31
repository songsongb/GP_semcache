#!/usr/bin/env bash
# Sourced by bounded Slurm blocks. This file never submits another job.
set -euo pipefail
: "${SLURM_JOB_ID:?manual Slurm allocation required}"
: "${I5_EXPERIMENT:?absolute prepared experiment directory required}"
: "${I5_MULTI_PREPARED:?verified expanded three-adapter preparation required}"
: "${I5_RUN_ROOT:?fresh I5 result parent required}"
: "${I5_SOURCE_COMMIT:?reviewed committed source identity required}"
[[ "${R5_I5_ENABLE_GPU:-0}" == 1 && "${R5_ENABLE_GPU:-0}" == 1 ]] || exit 2
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration ]] || exit 2
[[ "$(git rev-parse HEAD)" == "$I5_SOURCE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || exit 2
if [[ -n "${I5_CONDA_ROOT:-}" ]]; then source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"; else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$I5_RUN_ROOT"
i5_index="${SLURM_ARRAY_TASK_ID:-0}"
python - "$I5_EXPERIMENT" "$i5_expected_scale" "$i5_dataset_count" "$i5_index" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,'src')
from semcache.edgelora.r5_i5_evidence import verify
p=Path(sys.argv[1]);verify(p);m=json.loads((p/'i5_workload_manifest.json').read_text());index=int(sys.argv[4]);assert 0<=index<len(m['blocks']);b=m['blocks'][index]
assert (sys.argv[2]=='ANY' or m['scale']==sys.argv[2]) and not m['TEST_FIXTURE_ONLY']
assert sys.argv[3]=='ANY' or len({x['dataset'] for x in m['blocks']})==int(sys.argv[3])
assert b['resources']['expected_maximum_forwards']>0
PY
python -u scripts/119_run_r5_i5.py --experiment "$I5_EXPERIMENT" --block-index "$i5_index" \
  --multi-prepared "$I5_MULTI_PREPARED" --source-commit "$I5_SOURCE_COMMIT" \
  --enable-gpu-experiment --runtime-budget-seconds "${I5_RUNTIME_SECONDS:-9000}" \
  --output-dir "$I5_RUN_ROOT/block_${i5_index}_${SLURM_JOB_ID}"
