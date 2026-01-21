#!/usr/bin/env python3
"""
Aggregate per-atom SOAP matrices into fixed-length feature tables
for inner-average, max, and PCA pooling strategies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import wandb
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from soap_pipeline_clean.pooling import POOLING_FUNCS, load_feature_matrix


def load_labels(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["id"] = df["id"].str.strip()
    return df.set_index("id")["label"]


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def aggregate_method(
    manifest_df: pd.DataFrame,
    labels: pd.Series,
    feature_columns,
    soap_dir: Path,
    method: str,
    output_path: Path,
) -> None:
    pooling_fn = POOLING_FUNCS[method]
    rows = []
    ids = []

    for stem in tqdm(manifest_df["filename"], desc=f"Pooling ({method})"):
        csv_path = soap_dir / f"{stem}.csv"
        matrix = load_feature_matrix(csv_path, feature_columns)
        pooled = pooling_fn(matrix)
        rows.append(pooled)
        ids.append(stem)

    feature_df = pd.DataFrame(rows, columns=feature_columns, index=ids)
    feature_df.insert(0, "id", ids)
    feature_df["label"] = feature_df["id"].map(labels)
    missing = feature_df["label"].isna().sum()
    if missing:
        raise ValueError(f"Missing labels for {missing} structures in {method}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    print(f"Wrote {method} pooled dataset -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pooled SOAP datasets.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "soap_pipeline_clean/metadata/mof_manifest.csv",
        help="Path to the MOF manifest CSV.",
    )
    parser.add_argument(
        "--soap-dir",
        type=Path,
        default=ROOT / "soap_pipeline_clean/outputs/soap_2d",
        help="Directory containing per-MOF SOAP CSV files.",
    )
    parser.add_argument(
        "--soap-meta",
        type=Path,
        default=ROOT / "soap_pipeline_clean/outputs/soap_2d/manifest.json",
        help="JSON manifest describing feature columns.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "comb_id_labels.csv",
        help="CSV with id,label columns for all 6089 MOFs.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("inner", "max", "pca"),
        choices=sorted(POOLING_FUNCS.keys()),
        help="Pooling strategies to compute.",
    )
    parser.add_argument(
        "--include-ids",
        type=Path,
        default=None,
        help="Optional text file with one ID per line to include.",
    )
    parser.add_argument(
        "--exclude-ids",
        type=Path,
        default=None,
        help="Optional text file with one ID per line to exclude.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="full",
        help="Suffix for output files (e.g., trainval, test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "soap_pipeline_clean/outputs/pools",
        help="Directory to store aggregated datasets.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log pooling stats to W&B.")
    parser.add_argument("--wandb-project", type=str, default="mof-settransformer")
    parser.add_argument("--wandb-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_df = load_manifest(args.manifest)
    if args.include_ids:
        include_ids = {line.strip() for line in args.include_ids.read_text().splitlines() if line.strip()}
        manifest_df = manifest_df[manifest_df["filename"].isin(include_ids)]
    if args.exclude_ids:
        exclude_ids = {line.strip() for line in args.exclude_ids.read_text().splitlines() if line.strip()}
        manifest_df = manifest_df[~manifest_df["filename"].isin(exclude_ids)]
    manifest_df = manifest_df.reset_index(drop=True)

    if args.wandb:
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    labels = load_labels(args.labels)
    soap_meta = json.loads(args.soap_meta.read_text())
    feature_columns = soap_meta["feature_labels"]

    for method in args.methods:
        output_path = args.output_dir / f"{method}_{args.split_name}.csv"
        aggregate_method(
            manifest_df,
            labels,
            feature_columns,
            args.soap_dir,
            method,
            output_path,
        )
        if args.wandb:
            wandb.log(
                {
                    "method": method,
                    "split": args.split_name,
                    "rows": len(manifest_df),
                    "cols": len(feature_columns),
                }
            )

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
