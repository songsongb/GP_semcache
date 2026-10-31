#!/usr/bin/env bash
#SBATCH --job-name=r5_balanced
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodelist=moana-y4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5p_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5p_%j.err
set -euo pipefail
[[ "${R5P_ENABLE_GPU:-0}" == 1 && "${R5_ENABLE_GPU:-0}" == 1 ]] || { echo 'Both R5P_ENABLE_GPU=1 and R5_ENABLE_GPU=1 required.' >&2; exit 2; }
cd /data/khuss/repos/GP_semcache
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration ]] || exit 2
[[ -n "${R5P_SOURCE_COMMIT:-}" && "$(git rev-parse HEAD)" == "$R5P_SOURCE_COMMIT" ]] || { echo 'Set immutable R5P_SOURCE_COMMIT.' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" && -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || exit 2
[[ "${R5P_MODE:-diagnostic}" == diagnostic || "${R5P_MODE:-diagnostic}" == performance ]] || exit 2
if [[ -n "${R5P_CONDA_ROOT:-}" ]]; then source "$R5P_CONDA_ROOT/etc/profile.d/conda.sh";
else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TMPDIR="${SLURM_TMPDIR:-/tmp}"
export R5P_TINYBERT_REVISION=34707a33cd59a94ecde241ac209bf35103691b43
mkdir -p results/r5/performance
r5p_root="results/r5/performance/${R5P_MODE:-diagnostic}_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$r5p_root"
r5p_stage=preflight
trap 'r5p_status=$?; if [[ "$r5p_status" -ne 0 ]]; then printf "{\"exit\":%s,\"stage\":\"%s\",\"preserve_evidence\":true}\n" "$r5p_status" "$r5p_stage" > "$r5p_root/launcher_failure.json"; fi' EXIT
python -c 'import os,torch,numpy,transformers,peft; from transformers import AutoTokenizer,AutoModel; assert torch.cuda.is_available() and torch.cuda.device_count()==1 and "RTX 3090" in torch.cuda.get_device_name(0); AutoTokenizer.from_pretrained("facebook/opt-2.7b",local_files_only=True); AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D",revision=os.environ["R5P_TINYBERT_REVISION"],local_files_only=True); AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D",revision=os.environ["R5P_TINYBERT_REVISION"],local_files_only=True)'
export R5P_REPETITIONS="${R5P_REPETITIONS:-8}"
python -c 'import os,shutil; n=8 if os.environ.get("R5P_MODE","diagnostic")=="diagnostic" else 4*int(os.environ["R5P_REPETITIONS"]); assert shutil.disk_usage(".").free>(n+2)*2**30,"Insufficient worst-case block/evidence storage"'
r5p_stage=fresh_allocated_base_and_adapter
python -u scripts/95_run_r3a_distributed_no_reuse.py --phase prepare --model facebook/opt-2.7b --dtype float16 \
  --prompt-lengths 32 --device cuda:0 --adapter user_a --seed 42 --output-dir "$r5p_root/base_prepare"
r5p_stage=scan_attribution
python scripts/112_run_r5_balanced_benchmark.py --phase diagnose \
  --historical-inputs /data/khuss/repos/GP_semcache/results/r5/opt_152301_20261031T074321Z \
  --output-dir "$r5p_root/attribution"
if [[ "${R5P_MODE:-diagnostic}" == diagnostic ]]; then
  for r5p_accounting in legacy incremental; do
    r5p_extra=(); if [[ "$r5p_accounting" == legacy ]]; then r5p_extra+=(--legacy-scan); fi
    r5p_stage="strict_${r5p_accounting}_ABBA"
    python -u scripts/112_run_r5_balanced_benchmark.py --phase run --enable-gpu-experiment \
      --source-commit "$R5P_SOURCE_COMMIT" --base-prepared "$r5p_root/base_prepare" --semantic-revision "$R5P_TINYBERT_REVISION" \
      --repetitions 1 --start-order ABBA --generation-budget 4 --warmup-requests 1 --evidence-profile STRICT_AUDIT \
      --block-budget-bytes 1073741824 --reconcile-every 16 "${r5p_extra[@]}" --output-dir "$r5p_root/$r5p_accounting"
    python scripts/113_analyze_r5_balanced_benchmark.py --inputs "$r5p_root/$r5p_accounting" --output-dir "$r5p_root/${r5p_accounting}_analysis"
  done
else
  r5p_stage=balanced_scalable_evaluation
  python -u scripts/112_run_r5_balanced_benchmark.py --phase run --enable-gpu-experiment \
    --source-commit "$R5P_SOURCE_COMMIT" --base-prepared "$r5p_root/base_prepare" --semantic-revision "$R5P_TINYBERT_REVISION" \
    --repetitions "$R5P_REPETITIONS" --start-order ABBA --generation-budget 8 --warmup-requests 1 \
    --evidence-profile SCALABLE_PERFORMANCE --block-budget-bytes 1073741824 --reconcile-every 16 --output-dir "$r5p_root/run"
  python scripts/113_analyze_r5_balanced_benchmark.py --inputs "$r5p_root/run" --output-dir "$r5p_root/analysis"
fi
r5p_stage=complete
echo "R5-I3P complete: $r5p_root. Functional and performance gates must be read separately."
