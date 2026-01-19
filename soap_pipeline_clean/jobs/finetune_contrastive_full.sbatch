#!/bin/bash
#SBATCH -J st_finetune_full
#SBATCH -o outputs/logs/st_finetune_full_%j.out
#SBATCH -e outputs/logs/st_finetune_full_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --mem=64G

set -euo pipefail

mkdir -p outputs/logs outputs/set_transformer/finetune_full

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

python scripts/finetune_contrastive_regressor.py \
  --manifest metadata/mof_manifest.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --labels ../comb_id_labels.csv \
  --pretrained-encoder outputs/set_transformer/contrastive/encoder.pt \
  --output-dir outputs/set_transformer/finetune_full \
  --epochs 80 \
  --batch-size 8 \
  --latent-dim 510 \
  --train-frac 0.8 --val-frac 0.1 --test-frac 0.1 \
  --seed 42 \
  --original-only \
  --wandb \
  --wandb-project mof-settransformer \
  --wandb-name finetune_full_seed42
