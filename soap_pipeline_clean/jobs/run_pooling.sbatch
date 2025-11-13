#!/bin/bash
#SBATCH -J soap_pool
#SBATCH -o outputs/logs/pooling_%j.out
#SBATCH -e outputs/logs/pooling_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH --mem=32G

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
  --methods inner max pca \
  --output-dir outputs/pools
