#!/bin/bash
#SBATCH -J soap_pooled_bench
#SBATCH -o outputs/logs/soap_pooled_bench_%j.out
#SBATCH -e outputs/logs/soap_pooled_bench_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -p nvidia
#SBATCH -c 8
#SBATCH --time=08:00:00
#SBATCH --mem=32G

set -euo pipefail

mkdir -p outputs/logs outputs/pools

python scripts/run_pooled_benchmarks.py \
  --pools-dir outputs/pools \
  --methods inner max pca \
  --train-split trainval \
  --test-split test \
  --models ridge gbr rf svr xgb lgbm \
  --seed 42 \
  --output outputs/pools/benchmark_metrics.json \
  --wandb \
  --wandb-project mof-settransformer \
  --wandb-name pooled_benchmarks_6089
