#!/bin/bash
#SBATCH -J soap_pool_max
#SBATCH -o outputs/logs/soap_pool_max_%j.out
#SBATCH -e outputs/logs/soap_pool_max_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -p nvidia
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --mem=64G

set -euo pipefail

mkdir -p outputs/logs outputs/pools

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

python scripts/build_pooled_datasets.py \
  --manifest metadata/mof_manifest.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --labels ../comb_id_labels.csv \
  --methods max \
  --include-ids metadata/splits/trainval_ids.txt \
  --split-name trainval \
  --output-dir outputs/pools \
  --wandb \
  --wandb-project mof-settransformer \
  --wandb-name pools_max_trainval

python scripts/build_pooled_datasets.py \
  --manifest metadata/mof_manifest.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --labels ../comb_id_labels.csv \
  --methods max \
  --include-ids metadata/splits/test_ids.txt \
  --split-name test \
  --output-dir outputs/pools \
  --wandb \
  --wandb-project mof-settransformer \
  --wandb-name pools_max_test
