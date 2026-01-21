#!/bin/bash
#SBATCH -J soap_split
#SBATCH -o outputs/logs/soap_split_%j.out
#SBATCH -e outputs/logs/soap_split_%j.err
#SBATCH -D /scratch/mmm9886/Chignolin_Trajectory/SOAP_research/soap_pipeline_clean
#SBATCH -p nvidia
#SBATCH -c 2
#SBATCH --time=01:00:00
#SBATCH --mem=8G

set -euo pipefail

mkdir -p outputs/logs metadata/splits

python scripts/create_test_split.py --manifest metadata/mof_manifest.csv --test-frac 0.1 --seed 42 --output-dir metadata/splits
