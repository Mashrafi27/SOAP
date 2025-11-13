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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
from tqdm import tqdm

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
        "--output-dir",
        type=Path,
        default=ROOT / "soap_pipeline_clean/outputs/pools",
        help="Directory to store aggregated datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_df = load_manifest(args.manifest)
    labels = load_labels(args.labels)
    soap_meta = json.loads(args.soap_meta.read_text())
    feature_columns = soap_meta["feature_labels"]

    for method in args.methods:
        output_path = args.output_dir / f"{method}_full.csv"
        aggregate_method(
            manifest_df,
            labels,
            feature_columns,
            args.soap_dir,
            method,
            output_path,
        )


if __name__ == "__main__":
    main()
