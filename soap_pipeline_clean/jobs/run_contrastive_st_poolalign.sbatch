#!/bin/bash
#SBATCH -J st_poolalign_contrastive
#SBATCH -o outputs/logs/st_poolalign_contrastive_%j.out
#SBATCH -e outputs/logs/st_poolalign_contrastive_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --mem=64G

set -euo pipefail

mkdir -p outputs/logs outputs/set_transformer/contrastive_poolalign

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

python scripts/pretrain_contrastive_poolalign_settransformer.py \
  --manifest metadata/mof_manifest.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --labels ../comb_id_labels.csv \
  --pooled-dir outputs/pools \
  --pool-methods inner max pca \
  --pool-split trainval \
  --include-ids metadata/splits/trainval_ids.txt \
  --output-dir outputs/set_transformer/contrastive_poolalign \
  --epochs 80 \
  --batch-size 16 \
  --latent-dim 510 \
  --num-inds 32 \
  --dim-hidden 256 \
  --num-heads 4 \
  --wandb \
  --wandb-project mof-settransformer \
  --wandb-name contrastive_st_poolalign
