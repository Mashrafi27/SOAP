#!/bin/bash
#SBATCH -J soap_contrastive_mil
#SBATCH -o outputs/logs/soap_contrastive_mil_%j.out
#SBATCH -e outputs/logs/soap_contrastive_mil_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --mem=64G

set -euo pipefail

mkdir -p outputs/logs outputs/set_transformer/contrastive_mil

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

python scripts/pretrain_contrastive_mil.py \
  --manifest metadata/mof_manifest.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --labels ../comb_id_labels.csv \
  --pooled-dir outputs/pools \
  --pool-methods inner max pca \
  --pool-split trainval \
  --include-ids metadata/splits/trainval_ids.txt \
  --output-dir outputs/set_transformer/contrastive_mil \
  --epochs 80 \
  --batch-size 16 \
  --latent-dim 256 \
  --hidden-dim 256 \
  --wandb \
  --wandb-project mof-settransformer \
  --wandb-name contrastive_mil_poolalign
