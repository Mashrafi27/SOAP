#!/usr/bin/env python3
"""Benchmark pooled feature datasets with classical ML regressors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def load_pool(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = df["id"].str.strip()
    return df


def align_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_cols = [c for c in train_df.columns if c not in {"id", "label"}]
    test_cols = [c for c in test_df.columns if c not in {"id", "label"}]
    all_cols = sorted(set(train_cols) | set(test_cols))

    for col in all_cols:
        if col not in train_df:
            train_df[col] = 0.0
        if col not in test_df:
            test_df[col] = 0.0

    train_df = train_df[["id", "label"] + all_cols]
    test_df = test_df[["id", "label"] + all_cols]
    return train_df, test_df, all_cols


def build_model(name: str, seed: int):
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if name == "svr":
        return SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)
    if name == "gbr":
        return GradientBoostingRegressor(random_state=seed)
    if name == "rf":
        return RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=8)
    if name == "xgb":
        return XGBRegressor(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=8,
        )
    if name == "lgbm":
        return LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=8,
        )
    raise ValueError(f"Unknown model: {name}")


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pooled SOAP datasets.")
    parser.add_argument("--pools-dir", type=Path, default=ROOT / "soap_pipeline_clean/outputs/pools")
    parser.add_argument("--methods", nargs="+", default=["inner", "max", "pca"])
    parser.add_argument("--train-split", type=str, default="trainval")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--models", nargs="+", default=["ridge", "gbr", "rf", "svr", "xgb", "lgbm"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "soap_pipeline_clean/outputs/pools/benchmark_metrics.json")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="mof-settransformer")
    parser.add_argument("--wandb-name", type=str, default="pooled_benchmarks")
    args = parser.parse_args()

    if args.wandb:
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    results = {}
    for method in args.methods:
        train_path = args.pools_dir / f"{method}_{args.train_split}.csv"
        test_path = args.pools_dir / f"{method}_{args.test_split}.csv"
        train_df = load_pool(train_path)
        test_df = load_pool(test_path)
        train_df, test_df, feature_cols = align_features(train_df, test_df)

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        results[method] = {}
        for model_name in args.models:
            model = build_model(model_name, args.seed)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate(y_test, preds)
            metrics["n_train"] = int(len(train_df))
            metrics["n_test"] = int(len(test_df))
            metrics["n_features"] = int(len(feature_cols))
            results[method][model_name] = metrics
            print(method, model_name, metrics)
            if args.wandb:
                wandb.log({
                    "method": method,
                    "model": model_name,
                    "test_mae": metrics["mae"],
                    "test_rmse": metrics["rmse"],
                    "test_r2": metrics["r2"],
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
