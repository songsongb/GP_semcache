#!/usr/bin/env bash
#SBATCH --job-name=r5_semcache_e2e
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodelist=moana-y4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5_%j.err
set -euo pipefail
[[ "${R5_ENABLE_GPU:-0}" == 1 ]] || { echo 'Explicit R5_ENABLE_GPU=1 required.' >&2; exit 2; }
cd /data/khuss/repos/GP_semcache
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo 'Allocated RTX 3090 required.' >&2; exit 2; }
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration ]] || { echo 'Wrong R5 branch.' >&2; exit 2; }
[[ -n "${R5_SOURCE_COMMIT:-}" && "$(git rev-parse HEAD)" == "$R5_SOURCE_COMMIT" ]] || { echo 'Set R5_SOURCE_COMMIT to the committed implementation.' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Tracked source must be committed.' >&2; exit 2; }
[[ "${R5_TINYBERT_REVISION:-}" =~ ^[0-9a-f]{40}$ ]] || { echo 'Set immutable cached R5_TINYBERT_REVISION.' >&2; exit 2; }
mkdir -p results/r5
r5_root="results/r5/opt_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$r5_root"
r5_stage=environment
r5_on_exit() {
  r5_status=$?
  if [[ "$r5_status" -ne 0 ]]; then
    printf '{"status":"FAILED","exit":%s,"stage":"%s","preserve_partial_evidence":true}\n' \
      "$r5_status" "$r5_stage" > "$r5_root/launcher_failure.json"
  fi
}
trap r5_on_exit EXIT
if [[ -n "${R5_CONDA_ROOT:-}" ]]; then source "$R5_CONDA_ROOT/etc/profile.d/conda.sh";
else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TMPDIR="${SLURM_TMPDIR:-/tmp}"
r5_stage=offline_dependency_and_allocation_check
python -c 'import torch,numpy,transformers,peft; from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModel; import os; assert torch.cuda.is_available() and torch.cuda.device_count()==1 and "RTX 3090" in torch.cuda.get_device_name(0); AutoTokenizer.from_pretrained("facebook/opt-2.7b",local_files_only=True); AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D",revision=os.environ["R5_TINYBERT_REVISION"],local_files_only=True); AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D",revision=os.environ["R5_TINYBERT_REVISION"],local_files_only=True)'
python -c 'import shutil; assert shutil.disk_usage(".").free>6*2**30,"At least 6 GiB free disk required"'
r5_stage=fresh_allocated_base_adapter_preparation
# Existing isolated preparation records complete model/matrix fingerprints and this GPU UUID.
python -u scripts/95_run_r3a_distributed_no_reuse.py --phase prepare --model facebook/opt-2.7b --dtype float16 \
  --prompt-lengths 32 --device cuda:0 --adapter user_a --seed 42 --output-dir "$r5_root/base_prepare"
r5_stage=cpu_contract_checks
PYTHONPATH=src:tests python -m unittest tests.test_r5_contracts tests.test_r5_semantic tests.test_r5_controller -v > "$r5_root/cpu_contracts.log" 2>&1
r5_stage=integrated_real_opt_controlled_text
# Ten ordinary generated requests + ten actual no-reuse controls, G8.
# Capacity uses tokenizer-observed P/Q/R sizes; default 256 MiB remains supported.
python -u scripts/110_run_r5_semcache_e2e.py --phase run --enable-gpu-experiment \
  --source-commit "$R5_SOURCE_COMMIT" --semantic-revision "$R5_TINYBERT_REVISION" \
  --base-prepared "$r5_root/base_prepare" --lifecycle-capacity --generation-budget 8 \
  --evidence-budget-bytes 2147483648 --output-dir "$r5_root/run"
r5_stage=independent_readonly_audit
python scripts/111_analyze_r5_semcache_e2e.py --inputs "$r5_root/run" --output-dir "$r5_root/analysis"
r5_stage=complete
echo "R5 controlled integration complete: $r5_root; inspect distinct audit/fidelity/stage gates."
