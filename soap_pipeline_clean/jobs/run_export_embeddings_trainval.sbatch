#!/bin/bash
#SBATCH -J soap_export_trainval
#SBATCH -o outputs/logs/soap_export_trainval_%j.out
#SBATCH -e outputs/logs/soap_export_trainval_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -p nvidia
#SBATCH -c 4
#SBATCH --time=04:00:00
#SBATCH --mem=32G

set -euo pipefail

mkdir -p outputs/logs outputs/set_transformer/contrastive_full

python scripts/export_contrastive_embeddings.py --encoder outputs/set_transformer/contrastive_full/encoder.pt --manifest metadata/splits/trainval_manifest.csv --soap-dir outputs/soap_2d --soap-meta outputs/soap_2d/manifest.json --output outputs/set_transformer/contrastive_full/embeddings_trainval.csv
