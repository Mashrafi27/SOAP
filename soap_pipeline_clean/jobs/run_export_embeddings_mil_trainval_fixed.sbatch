#!/bin/bash
#SBATCH -J mil_export_trainval
#SBATCH -o outputs/logs/mil_export_trainval_%j.out
#SBATCH -e outputs/logs/mil_export_trainval_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -c 4
#SBATCH --time=06:00:00
#SBATCH --mem=32G

set -euo pipefail

mkdir -p outputs/logs outputs/pools

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/export_contrastive_embeddings_mil.py \
  --encoder outputs/set_transformer/contrastive_mil/encoder.pt \
  --manifest metadata/mof_manifest.csv \
  --labels ../comb_id_labels.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --include-ids metadata/splits/trainval_ids.txt \
  --output outputs/pools/mil_trainval.csv \
  --hidden-dim 512 \
  --latent-dim 256
