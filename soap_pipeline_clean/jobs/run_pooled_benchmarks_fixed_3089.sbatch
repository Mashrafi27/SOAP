#!/bin/bash
#SBATCH -J pooled_3089_fixed
#SBATCH -o outputs/logs/pooled_3089_fixed_%j.out
#SBATCH -e outputs/logs/pooled_3089_fixed_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -c 2
#SBATCH --time=04:00:00
#SBATCH --mem=32G

set -euo pipefail

mkdir -p outputs/logs outputs/pools

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/run_pooled_benchmarks.py \
  --pools-dir outputs/pools \
  --methods inner max pca \
  --train-split 3089_fixed \
  --test-split test_fixed \
  --models ridge gbr rf svr xgb lgbm \
  --seed 42 \
  --output outputs/pools/benchmark_metrics_3089_fixed.json \
  --wandb \
  --wandb-project mof-settransformer \
  --wandb-name pooled_3089_fixed
