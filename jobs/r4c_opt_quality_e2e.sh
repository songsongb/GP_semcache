#!/usr/bin/env bash
#SBATCH --job-name=r4c_opt_quality
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r4c_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r4c_%j.err
set -euo pipefail
[[ "${R4C_ENABLE_GPU:-0}" == 1 ]] || { echo 'Explicit R4C_ENABLE_GPU=1 required.' >&2; exit 2; }
cd /data/khuss/repos/GP_semcache
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo 'Allocated RTX3090 required.' >&2; exit 2; }
mkdir -p results/r4/c
r4c_root="results/r4/c/opt_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$r4c_root"
r4c_stage=branch_and_environment
r4c_on_exit() {
  r4c_status=$?
  if [[ "$r4c_status" -ne 0 ]]; then
    printf '{"status":"FAILED","exit":%s,"stage":"%s","preserve_partial_evidence":true}\n' \
      "$r4c_status" "$r4c_stage" > "$r4c_root/launcher_failure.json"
  fi
}
trap r4c_on_exit EXIT
[[ "$(git branch --show-current)" == exp/raw-semcache-r4c ]] || { echo 'Wrong work branch.' >&2; exit 2; }
[[ -n "${R4C_DATASET_REGISTRY:-}" ]] || { echo 'MISSING_DATASET: set R4C_DATASET_REGISTRY to a verified local registry.' >&2; exit 2; }
if [[ -n "${R4C_CONDA_ROOT:-}" ]]; then source "$R4C_CONDA_ROOT/etc/profile.d/conda.sh";
else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TMPDIR="${SLURM_TMPDIR:-/tmp}"
r4c_stage=preflight
python scripts/106_run_r4c_quality_e2e.py --phase dry-run --output-dir "$r4c_root/preflight"
r4c_stage=dataset_integrity
python scripts/105_prepare_r4c_longbench.py --validate-datasets-only --registry "$R4C_DATASET_REGISTRY" --output-dir "$r4c_root/dataset_check"
r4c_stage=dependency_and_gpu_check
python -c 'import importlib.util; names=("torch","numpy","transformers","peft","rouge","sacrebleu"); assert all(importlib.util.find_spec(n) for n in names), "Missing dependencies"; import torch; from importlib.metadata import version; assert version("rouge")=="1.0.1"; assert torch.cuda.device_count()==1 and "3090" in torch.cuda.get_device_name(0), "One RTX3090 required"'
r4c_stage=fresh_model_adapter_preparation
python -u scripts/95_run_r3a_distributed_no_reuse.py --phase prepare --model facebook/opt-2.7b --dtype float16 \
  --prompt-lengths 32 --device cuda:0 --adapter user_a --seed 42 --output-dir "$r4c_root/base_prepare"
r4c_stage=real_prompt_preparation
python scripts/105_prepare_r4c_longbench.py --registry "$R4C_DATASET_REGISTRY" --base-prepared "$r4c_root/base_prepare" \
  --count 2 --seed 42 --cohort development --input-tokens 256 --prefix-tokens 128 --qa-budget 32 --summary-budget 64 \
  --policy quality --strict-count 1 --output-dir "$r4c_root/prepared"
r4c_stage=cpu_contract_sanity
PYTHONPATH=src:tests python -m unittest tests.test_r4c_dataset tests.test_r4c_metrics tests.test_r4c_session tests.test_r4c_opt_contract -v > "$r4c_root/cpu_contracts.log" 2>&1
r4c_stage=physical_qa_and_summarization
python -u scripts/106_run_r4c_quality_e2e.py --phase run --enable-gpu-experiment --base-prepared "$r4c_root/base_prepare" \
  --prepared "$r4c_root/prepared" --capacity-bytes 268435456 --evidence-budget-bytes 2147483648 --output-dir "$r4c_root/run"
r4c_stage=independent_audit_gold_numerical_latency_analysis
python scripts/106_run_r4c_quality_e2e.py --phase analyze --inputs "$r4c_root/run" --output-dir "$r4c_root/analysis"
r4c_stage=complete
echo "C0/C1 diagnostic pilot complete: $r4c_root; inspect quality, strict fidelity and amortized costs separately."
