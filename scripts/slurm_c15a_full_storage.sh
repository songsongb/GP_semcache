#!/usr/bin/env bash
#SBATCH --job-name=c15a-storage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=c15a-storage-%j.out
set -euo pipefail

# Submit from the repository root with the existing semcache environment active.
# Supply account/partition via sbatch flags if required by the cluster.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
srun python scripts/40_cachegen_shared_cdf.py storage-full \
  --capture-manifest results/cachegen/c1/capture_manifest.json \
  --output-dir results/cachegen/c1_5 "$@"
