#!/bin/bash
#SBATCH -J soap_mil_benchmark
#SBATCH -o outputs/logs/soap_mil_benchmark_%j.out
#SBATCH -e outputs/logs/soap_mil_benchmark_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -c 8
#SBATCH --time=03:00:00
#SBATCH --mem=32G

set -euo pipefail

mkdir -p outputs/logs outputs/pools

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

python scripts/run_pooled_benchmarks.py \
  --pools-dir outputs/pools \
  --methods mil \
  --train-split trainval \
  --test-split test \
  --models ridge gbr rf svr xgb lgbm \
  --seed 42 \
  --output outputs/pools/benchmark_metrics_mil.json \
  --wandb \
  --wandb-project mof-settransformer \
  --wandb-name mil_benchmark
