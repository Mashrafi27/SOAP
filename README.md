# SOAP Research Workspace

This repository has two parallel workflows:

1. **Clean pipeline (`soap_pipeline_clean/`)** – fully scripted SOAP generation, pooling, and SetTransformer training.
2. **Legacy pipeline (`legacy_pipeline/`)** – historical notebooks, scripts, and derived CSVs preserved for reference. Everything previously in the repo (notebooks, csvs, hand-written scripts) now lives under `legacy_pipeline/` so the top level stays organized.

The rest of this README documents the clean pipeline since it’s where future development happens.

## Clean Pipeline Overview
- `soap_pipeline_clean/columns.py`: consistent SOAP feature labels.
- `soap_pipeline_clean/generator.py`: builds the per-atom 2D SOAP matrices (no averaging).
- `soap_pipeline_clean/scripts/compute_soap2d.py`: entry-point to generate descriptors for a CIF directory.
- `soap_pipeline_clean/scripts/build_manifest.py`: tags MOFs as original/new and assigns 500-sized batches.
- `soap_pipeline_clean/scripts/build_pooled_datasets.py`: derives inner/outer/max/PCA pools from the saved 2D matrices.
- `soap_pipeline_clean/pooling.py`: pooling helpers used by the CLI.
- `soap_pipeline_clean/set_transformer/*`: dataset and model definitions for the SetTransformer regressor + contrastive encoder.

## Environment Setup
```bash
module load --force purge
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda create -y -n soap_pipeline python=3.10
conda activate soap_pipeline
pip install -r soap_pipeline_clean/requirements.txt
```

## Batch Usage (recommended)
Generate per-atom SOAP matrices:
```bash
cd soap_pipeline_clean
sbatch jobs/run_soap2d.sbatch
squeue -u $USER
tail -f outputs/logs/soap2d_<JOBID>.out
```

Derive pooled datasets (inner/max/PCA):
```bash
sbatch jobs/run_pooling.sbatch
```

Train SetTransformer regressor (direct prediction):
```bash
sbatch jobs/train_set_transformer.sbatch
```

Contrastive pretrain the encoder (learned pooling):
```bash
sbatch jobs/pretrain_contrastive.sbatch
```

## Direct Invocation
```bash
python soap_pipeline_clean/scripts/compute_soap2d.py \
  --cif-dir comb_CIF_files \
  --output-dir soap_pipeline_clean/outputs/soap_2d \
  --r-cut 5.0 --n-max 1 --l-max 1 --sigma 0.5 --n-jobs 1
```

## Notes
- Outputs and logs live under `soap_pipeline_clean/outputs/`.
- `legacy_pipeline/` holds the historical notebooks (`KernelPCA.ipynb`, `MOF_SOAP_ML.ipynb`, etc.) plus the aggregated CSVs (`inner_averaged_local_soap_mofs.csv`, etc.) so nothing was lost. When in doubt, look there.
