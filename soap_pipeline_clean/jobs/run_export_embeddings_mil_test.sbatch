#!/bin/bash
#SBATCH -J soap_mil_embed_test
#SBATCH -o outputs/logs/soap_mil_embed_test_%j.out
#SBATCH -e outputs/logs/soap_mil_embed_test_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -c 8
#SBATCH --time=06:00:00
#SBATCH --mem=32G

set -euo pipefail

mkdir -p outputs/logs outputs/pools

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

python scripts/export_contrastive_embeddings_mil.py \
  --encoder outputs/set_transformer/contrastive_mil/encoder.pt \
  --manifest metadata/mof_manifest.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --include-ids metadata/splits/test_ids.txt \
  --output outputs/pools/mil_test.csv
