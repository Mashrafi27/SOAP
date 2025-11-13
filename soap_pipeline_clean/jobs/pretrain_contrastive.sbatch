#!/bin/bash
#SBATCH -J st_contrast
#SBATCH -o outputs/logs/st_contrast_%j.out
#SBATCH -e outputs/logs/st_contrast_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --mem=64G

set -euo pipefail

mkdir -p outputs/logs outputs/set_transformer/contrastive

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

python scripts/pretrain_contrastive.py \
  --manifest metadata/mof_manifest.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --labels ../comb_id_labels.csv \
  --output-dir outputs/set_transformer/contrastive \
  --epochs 80 \
  --batch-size 16 \
  --atom-drop 0.3 \
  --feature-drop 0.15 \
  --latent-dim 510
