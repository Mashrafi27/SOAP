#!/bin/bash
#SBATCH -J soap_automl
#SBATCH -o outputs/logs/automl_%j.out
#SBATCH -e outputs/logs/automl_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -c 4
#SBATCH --time=12:00:00
#SBATCH --mem=16G

set -euo pipefail

mkdir -p outputs/logs outputs/automl

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

PYTHON_BIN="$HOME/.conda/envs/soap_analysis/bin/python"

"${PYTHON_BIN}" scripts/run_automl_subsets.py \
  --manifest metadata/mof_manifest.csv \
  --pools-dir outputs/pools \
  --methods inner_full.csv max_full.csv pca_full.csv \
  --output outputs/automl/results.json
