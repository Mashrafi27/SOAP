#!/bin/bash
#SBATCH -J soap2d
#SBATCH -o soap_pipeline_clean/outputs/logs/soap2d_%j.out
#SBATCH -e soap_pipeline_clean/outputs/logs/soap2d_%j.err
#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --mem=16G

set -euo pipefail

# Ensure conda is available (adapt this to your cluster setup if needed)
CONDA_BASE="/share/apps/NYUAD5/miniconda/3-4.11.0"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_NAME="soap_pipeline"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[WARN] Conda environment '${ENV_NAME}' not found."
  echo "       Create it once with:"
  echo "         conda create -y -n ${ENV_NAME} python=3.10"
  echo "         conda activate ${ENV_NAME}"
  echo "         pip install -r soap_pipeline_clean/requirements.txt"
  exit 1
fi

conda activate "${ENV_NAME}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CIF_DIR="comb_CIF_files"
OUTPUT_DIR="soap_pipeline_clean/outputs/soap_2d"

mkdir -p "${OUTPUT_DIR}"

python soap_pipeline_clean/scripts/compute_soap2d.py \
  --cif-dir "${CIF_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --r-cut 5.0 \
  --n-max 1 \
  --l-max 1 \
  --sigma 0.5 \
  --n-jobs 1 \
  --file-format csv
