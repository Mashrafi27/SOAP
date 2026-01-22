#!/bin/bash
#SBATCH -J soap_pooled_bench_6089
#SBATCH -o outputs/logs/soap_pooled_bench_6089_%j.out
#SBATCH -e outputs/logs/soap_pooled_bench_6089_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -p nvidia
#SBATCH -c 8
#SBATCH --time=08:00:00
#SBATCH --mem=32G

set -euo pipefail

mkdir -p outputs/logs outputs/pools metadata/splits_6089

# Create split for full set (6089)
python scripts/create_test_split.py --manifest metadata/mof_manifest.csv --test-frac 0.1 --seed 42 --origin all --output-dir metadata/splits_6089

# Pool train/val and test for full set
python scripts/build_pooled_datasets.py --manifest metadata/mof_manifest.csv --soap-dir outputs/soap_2d --soap-meta outputs/soap_2d/manifest.json --labels ../comb_id_labels.csv --methods inner max pca --include-ids metadata/splits_6089/trainval_ids.txt --split-name trainval_6089 --output-dir outputs/pools --wandb --wandb-project mof-settransformer --wandb-name pools_trainval_6089

python scripts/build_pooled_datasets.py --manifest metadata/mof_manifest.csv --soap-dir outputs/soap_2d --soap-meta outputs/soap_2d/manifest.json --labels ../comb_id_labels.csv --methods inner max pca --include-ids metadata/splits_6089/test_ids.txt --split-name test_6089 --output-dir outputs/pools --wandb --wandb-project mof-settransformer --wandb-name pools_test_6089

# Benchmark
python scripts/run_pooled_benchmarks.py --pools-dir outputs/pools --methods inner max pca --train-split trainval_6089 --test-split test_6089 --models ridge gbr rf svr xgb lgbm --seed 42 --output outputs/pools/benchmark_metrics_6089.json --wandb --wandb-project mof-settransformer --wandb-name pooled_benchmarks_6089
