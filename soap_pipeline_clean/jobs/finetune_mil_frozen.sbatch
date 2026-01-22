#!/bin/bash
#SBATCH -J mil_finetune_frozen
#SBATCH -o outputs/logs/mil_finetune_frozen_%j.out
#SBATCH -e outputs/logs/mil_finetune_frozen_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --mem=64G

set -euo pipefail

mkdir -p outputs/logs outputs/set_transformer/finetune_mil_frozen

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

python scripts/finetune_mil_regressor.py \
  --manifest metadata/mof_manifest.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --labels ../comb_id_labels.csv \
  --pretrained-encoder outputs/set_transformer/contrastive_mil/encoder.pt \
  --output-dir outputs/set_transformer/finetune_mil_frozen \
  --freeze-encoder \
  --epochs 80 \
  --batch-size 16 \
  --latent-dim 256 \
  --hidden-dim 256 \
  --predictor-hidden 256 \
  --train-ids metadata/splits/trainval_ids.txt \
  --test-ids metadata/splits/test_ids.txt \
  --val-frac 0.1 \
  --seed 42 \
  --wandb \
  --wandb-project mof-settransformer \
  --wandb-name finetune_mil_frozen
