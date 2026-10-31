#!/usr/bin/env bash
#SBATCH --job-name=r5_fidelity
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodelist=moana-y4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5f_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5f_%j.err
set -euo pipefail
[[ "${R5F_ENABLE_GPU:-0}" == 1 && "${R5_ENABLE_GPU:-0}" == 1 ]] || { echo 'R5F_ENABLE_GPU=1 and R5_ENABLE_GPU=1 required.' >&2; exit 2; }
cd /data/khuss/repos/GP_semcache
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration ]] || exit 2
[[ -n "${R5F_SOURCE_COMMIT:-}" && "$(git rev-parse HEAD)" == "$R5F_SOURCE_COMMIT" ]] || { echo 'Set committed R5F_SOURCE_COMMIT.' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" && -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || exit 2
if [[ -n "${R5F_CONDA_ROOT:-}" ]]; then source "$R5F_CONDA_ROOT/etc/profile.d/conda.sh";
else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TMPDIR="${SLURM_TMPDIR:-/tmp}"
export R5F_TINYBERT_REVISION=34707a33cd59a94ecde241ac209bf35103691b43
mkdir -p results/r5/fidelity
r5f_root="results/r5/fidelity/opt_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$r5f_root"
r5f_stage=preflight
trap 'r5f_status=$?; if [[ "$r5f_status" -ne 0 ]]; then printf "{\"exit\":%s,\"stage\":\"%s\",\"preserve_evidence\":true}\n" "$r5f_status" "$r5f_stage" > "$r5f_root/launcher_failure.json"; fi' EXIT
python -c 'import os,shutil,torch,numpy,transformers,peft; from transformers import AutoTokenizer,AutoModel; assert shutil.disk_usage(".").free>10*2**30; assert torch.cuda.is_available() and torch.cuda.device_count()==1 and "RTX 3090" in torch.cuda.get_device_name(0); AutoTokenizer.from_pretrained("facebook/opt-2.7b",local_files_only=True); AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D",revision=os.environ["R5F_TINYBERT_REVISION"],local_files_only=True); AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D",revision=os.environ["R5F_TINYBERT_REVISION"],local_files_only=True)'
r5f_stage=fresh_allocated_base_and_adapter
python -u scripts/95_run_r3a_distributed_no_reuse.py --phase prepare --model facebook/opt-2.7b --dtype float16 \
  --prompt-lengths 32 --device cuda:0 --adapter user_a --seed 42 --output-dir "$r5f_root/base_prepare"
r5f_stage=forensic_capture
python -u scripts/114_run_r5_fidelity_forensics.py --phase run --enable-gpu-experiment \
  --source-commit "$R5F_SOURCE_COMMIT" --base-prepared "$r5f_root/base_prepare" \
  --semantic-revision "$R5F_TINYBERT_REVISION" --generation-budget 8 --block-budget-bytes 4294967296 \
  --output-dir "$r5f_root/run"
r5f_stage=independent_analysis
python scripts/115_analyze_r5_fidelity_forensics.py --inputs "$r5f_root/run" --output-dir "$r5f_root/analysis"
r5f_stage=complete
echo "R5-I3F captured: $r5f_root. Read fidelity_decision.json and fidelity_divergence.json; completion is not numerical parity."
