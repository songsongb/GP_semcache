#!/usr/bin/env bash
#SBATCH --job-name=c15b2-storage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=c15b2-storage-%j.out
set -euo pipefail

# Submit from the repository root with the existing C1 environment active.
# Fit profiles and complete smoke first. Supply account/partition externally.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
srun python scripts/42_cachegen_anchor_storage.py storage-full \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --output-dir results/cachegen/c1_5b/b2 "$@"
