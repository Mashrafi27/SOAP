#!/bin/bash
#SBATCH -J soap_pooled_bench_3089
#SBATCH -o outputs/logs/soap_pooled_bench_3089_%j.out
#SBATCH -e outputs/logs/soap_pooled_bench_3089_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -p nvidia
#SBATCH -c 8
#SBATCH --time=08:00:00
#SBATCH --mem=32G

set -euo pipefail

mkdir -p outputs/logs outputs/pools metadata/splits_3089

# Create split for original-only set (3089)
python scripts/create_test_split.py --manifest metadata/mof_manifest.csv --test-frac 0.1 --seed 42 --origin original --output-dir metadata/splits_3089

# Pool train/val and test for original-only
python scripts/build_pooled_datasets.py --manifest metadata/mof_manifest.csv --soap-dir outputs/soap_2d --soap-meta outputs/soap_2d/manifest.json --labels ../comb_id_labels.csv --methods inner max pca --include-ids metadata/splits_3089/trainval_ids.txt --split-name trainval_3089 --output-dir outputs/pools --wandb --wandb-project mof-settransformer --wandb-name pools_trainval_3089

python scripts/build_pooled_datasets.py --manifest metadata/mof_manifest.csv --soap-dir outputs/soap_2d --soap-meta outputs/soap_2d/manifest.json --labels ../comb_id_labels.csv --methods inner max pca --include-ids metadata/splits_3089/test_ids.txt --split-name test_3089 --output-dir outputs/pools --wandb --wandb-project mof-settransformer --wandb-name pools_test_3089

# Benchmark
python scripts/run_pooled_benchmarks.py --pools-dir outputs/pools --methods inner max pca --train-split trainval_3089 --test-split test_3089 --models ridge gbr rf svr xgb lgbm --seed 42 --output outputs/pools/benchmark_metrics_3089.json --wandb --wandb-project mof-settransformer --wandb-name pooled_benchmarks_3089
