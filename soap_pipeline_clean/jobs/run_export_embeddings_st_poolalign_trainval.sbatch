#!/bin/bash
#SBATCH -J st_poolalign_export_trainval
#SBATCH -o outputs/logs/st_poolalign_export_trainval_%j.out
#SBATCH -e outputs/logs/st_poolalign_export_trainval_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -c 4
#SBATCH --time=04:00:00
#SBATCH --mem=32G

set -euo pipefail

mkdir -p outputs/logs outputs/pools

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

python scripts/export_contrastive_embeddings.py \
  --encoder outputs/set_transformer/contrastive_poolalign/encoder.pt \
  --manifest metadata/mof_manifest.csv \
  --include-ids metadata/splits/trainval_ids.txt \
  --labels ../comb_id_labels.csv \
  --soap-dir outputs/soap_2d \
  --soap-meta outputs/soap_2d/manifest.json \
  --output outputs/pools/st_poolalign_trainval.csv \
  --latent-dim 510 \
  --num-inds 32 \
  --dim-hidden 256 \
  --num-heads 4
