# SOAP Research Workspace

This repository now contains two parallel workflows:

1. **Clean pipeline (`soap_pipeline_clean/`)** – the new, fully scriptable SOAP generation + pooling + SetTransformer training stack described below.
2. **Legacy pipeline (`legacy_pipeline/`)** – the historical scripts and notebooks (`avg_soap_generating_script.py`, `soap.py`, kernel-PCA variants, etc.) preserved for reference. They still run exactly as before, but all development going forward should happen in the clean pipeline.

The sections that follow document the clean pipeline, since that is the workflow we actively maintain.

## What’s Included
- `columns.py`: Helpers for producing consistent SOAP feature labels from a configured descriptor.
- `generator.py`: Core logic to discover CIF files, configure the SOAP descriptor, and export per‑structure 2D SOAP tensors (no averaging).
- `scripts/compute_soap2d.py`: Command line entry point that ties everything together.

## Environment Setup
Create a dedicated conda environment once and install the required packages:
```bash
module load --force purge  # if your site uses modules (optional)
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda create -y -n soap_pipeline python=3.10
conda activate soap_pipeline
pip install -r soap_pipeline_clean/requirements.txt
```

## Batch Usage (recommended)
Submit the preconfigured SLURM script to generate SOAP descriptors for every CIF in `comb_CIF_files`:
```bash
sbatch soap_pipeline_clean/jobs/run_soap2d.sbatch
# monitor
squeue -u $USER
tail -f soap_pipeline_clean/outputs/logs/soap2d_<JOBID>.out
```

Logs are written to `soap_pipeline_clean/outputs/logs/`. Each job writes its results to `soap_pipeline_clean/outputs/soap_2d` and a summary manifest.

## Direct Invocation (optional)
If you need to run the generator interactively from the activated environment:
```bash
python soap_pipeline_clean/scripts/compute_soap2d.py \
  --cif-dir comb_CIF_files \
  --output-dir soap_pipeline_clean/outputs/soap_2d \
  --r-cut 5.0 \
  --n-max 1 \
  --l-max 1 \
  --sigma 0.5 \
  --n-jobs 1
```

## Workflow Summary
The script will:
1. Scan the CIF directory to determine the union of chemical species.
2. Instantiate a `SOAP` descriptor with `average='off'` (full 2D tensor).
3. Generate one file per MOF in the output directory (CSV by default) containing atom metadata and SOAP features.
4. Emit a manifest (`manifest.json`) summarizing processed files, descriptor hyperparameters, and feature layout for downstream use.

## Notes
- Dependencies are locked in `requirements.txt` for reproducibility.
- The output directory is created automatically. Existing files with the same name are overwritten.
- Use `--file-format parquet` to save compact columnar files if `pyarrow` is available.

## Derived datasets
- `scripts/build_manifest.py` creates `metadata/mof_manifest.csv`, tagging original vs new MOFs and assigning 500-sized batches.
- `jobs/run_pooling.sbatch` (or `scripts/build_pooled_datasets.py`) converts the per-atom SOAP matrices into pooled tables (`inner`, `max`, `pca`) used by downstream AutoML.

## SetTransformer pipelines
- `scripts/train_set_transformer.py` + `jobs/train_set_transformer.sbatch` reproduce the end-to-end SetTransformer regressor directly on the 2D SOAP matrices (3k or full 6k).
- `scripts/pretrain_contrastive.py` + `jobs/pretrain_contrastive.sbatch` perform SimCLR-style contrastive pretraining to learn 510-d latent embeddings that act as a learned pooling strategy. The script also exports the embeddings for downstream tabular models.

All logs land in `outputs/logs/`, checkpoints/metrics in `outputs/set_transformer/`.
