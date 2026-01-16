#!/usr/bin/env python3
"""AutoML driver for pooled SOAP datasets with 500-increment cohorts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["filename"] = df["filename"].str.strip()
    return df


def load_pooled(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = df["id"].str.strip()
    return df


def build_subset(manifest: pd.DataFrame, pooled: pd.DataFrame, size: int) -> pd.DataFrame:
    selected = manifest.head(size)["filename"]
    subset = pooled[pooled["id"].isin(selected)].copy()
    if len(subset) != len(selected):
        missing = set(selected) - set(subset["id"])
        raise ValueError(f"Missing {len(missing)} IDs in pooled dataset: {list(missing)[:5]}")
    subset = subset.sort_values("id")
    feature_cols = [c for c in subset.columns if c not in {"id", "label"}]
    zero_cols = subset[feature_cols].columns[(subset[feature_cols] == 0).all()]
    subset = subset.drop(columns=zero_cols)
    return subset


def run_automl(subset: pd.DataFrame, seed: int) -> dict:
    feature_cols = [c for c in subset.columns if c not in {"id", "label"}]
    X = subset[feature_cols].values
    y = subset["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=8, max_depth=None)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "r2": float(r2_score(y_test, preds)),
        "n_features": len(feature_cols),
        "n_samples": len(subset),
    }


def main():
    parser = argparse.ArgumentParser(description="Run AutoML on pooled subsets.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "soap_pipeline_clean/metadata/mof_manifest.csv")
    parser.add_argument("--pools-dir", type=Path, default=ROOT / "soap_pipeline_clean/outputs/pools")
    parser.add_argument("--methods", nargs="+", default=("inner_full.csv", "max_full.csv", "pca_full.csv"))
    parser.add_argument("--base-size", type=int, default=3089, help="Number of original MOFs (from CIF_files) to include in every subset.")
    parser.add_argument(
        "--new-increments",
        nargs="+",
        type=int,
        default=[0, 500, 1000, 1500, 2000, 2500, 3000],
        help="Number of new MOFs (from the extra 3k) to add on top of the base set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "soap_pipeline_clean/outputs/automl/results.json")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    base_df = manifest[manifest["origin"] == "original"].sort_values("combined_rank")
    base_ids = base_df.head(args.base_size)["filename"].tolist()
    if len(base_ids) < args.base_size:
        raise ValueError(f"Requested base size {args.base_size} but only found {len(base_ids)} original entries.")
    new_df = manifest[manifest["origin"] == "new"].sort_values("new_index")
    new_ids_ordered = new_df["filename"].tolist()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for method in args.methods:
        pooled = load_pooled(args.pools_dir / method)
        method_key = Path(method).stem
        results[method_key] = {}
        for n_new in args.new_increments:
            n_new = min(n_new, len(new_ids_ordered))
            selected_ids = base_ids + new_ids_ordered[:n_new]
            subset = pooled[pooled["id"].isin(selected_ids)].copy()
            subset = subset.sort_values("id")
            feature_cols = [c for c in subset.columns if c not in {"id", "label"}]
            zero_cols = subset[feature_cols].columns[(subset[feature_cols] == 0).all()]
            subset = subset.drop(columns=zero_cols)
            metrics = run_automl(subset, args.seed)
            sample_size = len(subset)
            key = f"{sample_size}"
            results[method_key][key] = metrics
            results[method_key][key]["n_samples"] = sample_size
            results[method_key][key]["n_features"] = len([c for c in subset.columns if c not in {"id", "label"}])
            print(method_key, sample_size, metrics)

    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
