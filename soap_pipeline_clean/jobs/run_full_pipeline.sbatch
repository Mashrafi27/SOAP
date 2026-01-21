#!/bin/bash
#SBATCH -J soap_full_pipeline
#SBATCH -o outputs/logs/soap_full_pipeline_%j.out
#SBATCH -e outputs/logs/soap_full_pipeline_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=48:00:00
#SBATCH --mem=64G

set -euo pipefail

mkdir -p outputs/logs outputs/pools outputs/set_transformer/contrastive_full metadata/splits

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 1) Create fixed train/test split
python scripts/create_test_split.py \
  --manifest metadata/mof_manifest.csv \
  --test-frac 0.1 \
  --seed 42 \
  --output-dir metadata/splits

# 2) Pooling datasets (train/val only and test held-out)
python scripts/build_pooled_datasets.py \
  --manifest metadata/mof_manifest.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --labels ../comb_id_labels.csv \
  --methods inner max pca \
  --include-ids metadata/splits/trainval_ids.txt \
  --split-name trainval \
  --output-dir outputs/pools \
  --wandb \
  --wandb-project mof-settransformer \
  --wandb-name pools_trainval

python scripts/build_pooled_datasets.py \
  --manifest metadata/mof_manifest.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --labels ../comb_id_labels.csv \
  --methods inner max pca \
  --include-ids metadata/splits/test_ids.txt \
  --split-name test \
  --output-dir outputs/pools \
  --wandb \
  --wandb-project mof-settransformer \
  --wandb-name pools_test

# 3) Contrastive pretraining on all non-test data
python scripts/pretrain_contrastive.py \
  --manifest metadata/mof_manifest.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --labels ../comb_id_labels.csv \
  --output-dir outputs/set_transformer/contrastive_full \
  --use-all \
  --test-ids metadata/splits/test_ids.txt \
  --epochs 80 \
  --batch-size 16 \
  --latent-dim 510

# 4) Export embeddings for train/val and test sets
python scripts/export_contrastive_embeddings.py \
  --encoder outputs/set_transformer/contrastive_full/encoder.pt \
  --manifest metadata/splits/trainval_manifest.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --output outputs/set_transformer/contrastive_full/embeddings_trainval.csv

python scripts/export_contrastive_embeddings.py \
  --encoder outputs/set_transformer/contrastive_full/encoder.pt \
  --manifest metadata/splits/test_manifest.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --output outputs/set_transformer/contrastive_full/embeddings_test.csv
