#!/usr/bin/env bash
#SBATCH --job-name=r5_i5_smoke_one
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodelist=moana-y4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/khuss/repos/GP_semcache
#SBATCH --output=/data/khuss/repos/GP_semcache/logs/r5_i5_smoke_one_%A_%a.out
#SBATCH --error=/data/khuss/repos/GP_semcache/logs/r5_i5_smoke_one_%A_%a.err
# One independent A/B/C block per manually selected array task; use %1.
i5_expected_scale=SMOKE
i5_dataset_count=1
source jobs/r5_i5_common.sh
