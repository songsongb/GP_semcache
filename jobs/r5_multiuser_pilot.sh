#!/usr/bin/env bash
#SBATCH --job-name=r5_multi_pilot
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodelist=moana-y4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5m_pilot_%j.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5m_pilot_%j.err
set -euo pipefail
[[ "${R5M_ENABLE_GPU:-0}" == 1 && "${R5_ENABLE_GPU:-0}" == 1 && -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || exit 2
cd /data/khuss/repos/GP_semcache
[[ "$(git branch --show-current)" == exp/raw-semcache-r5-integration && -n "${R5M_SOURCE_COMMIT:-}" && "$(git rev-parse HEAD)" == "$R5M_SOURCE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || exit 2
if [[ -n "${R5M_CONDA_ROOT:-}" ]]; then source "$R5M_CONDA_ROOT/etc/profile.d/conda.sh"; else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p results/r5/multiuser
r5m_root="results/r5/multiuser/pilot_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$r5m_root"
r5m_stage=preflight
trap 'r5m_status=$?; if [[ "$r5m_status" -ne 0 ]]; then printf "{\"exit\":%s,\"stage\":\"%s\",\"preserve_evidence\":true}\n" "$r5m_status" "$r5m_stage" > "$r5m_root/launcher_failure.json"; fi' EXIT
python -c 'import os,shutil,torch,numpy,transformers,peft; from transformers import AutoTokenizer,AutoModel; assert shutil.disk_usage(".").free>10*2**30; assert torch.cuda.is_available() and torch.cuda.device_count()==1 and "RTX 3090" in torch.cuda.get_device_name(0); assert torch.cuda.mem_get_info(0)[0]>8*2**30; AutoTokenizer.from_pretrained("facebook/opt-2.7b",local_files_only=True); AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D",revision="34707a33cd59a94ecde241ac209bf35103691b43",local_files_only=True)'
r5m_stage=base_preparation
python -u scripts/95_run_r3a_distributed_no_reuse.py --phase prepare --model facebook/opt-2.7b --dtype float16 --prompt-lengths 32 --device cuda:0 --adapter user_a --seed 42 --output-dir "$r5m_root/base_prepare"
r5m_stage=distinct_adapter_preparation
python scripts/116_run_r5_multiuser.py --phase prepare --base-prepared "$r5m_root/base_prepare" --output-dir "$r5m_root/adapters"
r5m_stage=sequential_generation
python -u scripts/116_run_r5_multiuser.py --phase run --enable-gpu-experiment --stage pilot --multi-prepared "$r5m_root/adapters" --source-commit "$R5M_SOURCE_COMMIT" --generation-budget 4 --output-dir "$r5m_root/run"
r5m_stage=independent_audit
python scripts/117_analyze_r5_multiuser.py --inputs "$r5m_root/run" --output-dir "$r5m_root/analysis"
r5m_stage=complete
