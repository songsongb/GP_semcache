#!/usr/bin/env bash
#SBATCH --job-name=r4a_opt_e2e
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=logs/r4a_opt_e2e_%j.out
#SBATCH --error=logs/r4a_opt_e2e_%j.err
set -euo pipefail
# mkdir -p logs before submission: Slurm opens logs before this script starts.
cd "${R4A_REPO:-/data/khuss/repos/GP_semcache}"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo 'One allocated GPU required.' >&2; exit 2; }
mkdir -p results/r4/a
r4a_root="results/r4/a/opt_full_e2e_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$r4a_root"
trap 'r4a_status=$?; if [[ "$r4a_status" -ne 0 ]]; then printf "FAILED exit=%s; preserve logs and all partial evidence.\n" "$r4a_status" > "$r4a_root/launcher_failure.txt"; fi' EXIT
if [[ -n "${R4A_CONDA_ROOT:-}" ]]; then source "$R4A_CONDA_ROOT/etc/profile.d/conda.sh";
else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HOME=/data/khuss/huggingface HF_HUB_CACHE=/data/khuss/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TMPDIR="${SLURM_TMPDIR:-/tmp}"
echo "R4-A host=$(hostname) allocation=$SLURM_JOB_ID visible_gpu=$CUDA_VISIBLE_DEVICES output=$r4a_root"
git branch --show-current
git rev-parse HEAD
git status --short
python scripts/102_run_r4a_opt_full_e2e.py --phase dry-run --output-dir "${r4a_root}_preflight"
python -c 'import importlib.util; names=("torch","numpy","transformers","peft","sacrebleu"); missing=[n for n in names if importlib.util.find_spec(n) is None]; assert not missing, "MISSING_DEPENDENCY: "+str(missing)'
python -u scripts/95_run_r3a_distributed_no_reuse.py --phase prepare --model facebook/opt-2.7b --dtype float16 \
    --prompt-lengths 32 --device cuda:0 --adapter user_a --seed 42 --output-dir "$r4a_root/base_prepare"
python -u scripts/102_run_r4a_opt_full_e2e.py --phase prepare --enable-gpu-experiment \
    --base-prepared "$r4a_root/base_prepare" --scope "${R4A_SCOPE:-matrix}" --generation-policy fixed --output-dir "$r4a_root/prepare"
python -u scripts/102_run_r4a_opt_full_e2e.py --phase run --enable-gpu-experiment \
    --prepared "$r4a_root/prepare/native" --output-dir "$r4a_root/run"
python scripts/102_run_r4a_opt_full_e2e.py --phase analyze --inputs "$r4a_root/run" --output-dir "$r4a_root/aggregate"
echo "R4-A diagnostic complete; Native FAIL/unfavorable or unresolved latency are valid: $r4a_root"
