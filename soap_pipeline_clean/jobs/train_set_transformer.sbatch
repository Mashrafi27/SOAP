#!/bin/bash
#SBATCH -J st_reg
#SBATCH -o outputs/logs/st_reg_%j.out
#SBATCH -e outputs/logs/st_reg_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --mem=64G

set -euo pipefail

mkdir -p outputs/logs outputs/set_transformer/regression

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

python scripts/train_set_transformer.py \
  --manifest metadata/mof_manifest.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --labels ../comb_id_labels.csv \
  --output-dir outputs/set_transformer/regression \
  --epochs 60 \
  --batch-size 8 \
  --latent-dim 128
