#!/usr/bin/env bash
#SBATCH --job-name=r5_i5_analysis
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_i5_analysis_%j.out
set -euo pipefail
: "${I5_EXPERIMENT:?required}" "${I5_RUN_ROOT:?required}" "${I5_ANALYSIS_OUT:?fresh output required}"
if [[ -n "${I5_CONDA_ROOT:-}" ]]; then source "$I5_CONDA_ROOT/etc/profile.d/conda.sh"; else source "$(conda info --base)/etc/profile.d/conda.sh"; fi
conda activate semcache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
i5_runs=()
while IFS= read -r -d '' i5_status; do i5_runs+=("${i5_status%/i5_block_status.json}"); done < <(find "$I5_RUN_ROOT" -mindepth 2 -maxdepth 2 -name i5_block_status.json -print0)
[[ ${#i5_runs[@]} -gt 0 ]] || exit 2
python scripts/121_summarize_r5_i5.py --experiment "$I5_EXPERIMENT" --runs "${i5_runs[@]}" --output-dir "$I5_ANALYSIS_OUT"
python scripts/122_plot_r5_i5.py --inputs "$I5_ANALYSIS_OUT" --output-dir "${I5_ANALYSIS_OUT}_plots"
