#!/usr/bin/env python3
"""Train regressors on contrastive embeddings and report metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def load_embeddings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = df["id"].str.strip()
    return df


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = df["id"].str.strip()
    return df


def build_dataset(embeddings: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = embeddings.merge(labels, on="id", how="inner")
    missing = set(embeddings["id"]) - set(merged["id"])
    if missing:
        print(f"Warning: {len(missing)} embeddings missing labels (e.g. {list(missing)[:5]})")
    return merged


def build_model(model: str, seed: int):
    if model == "ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if model == "gbr":
        return GradientBoostingRegressor(random_state=seed)
    if model == "svr":
        return SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)
    if model == "xgb":
        return XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=8,
        )
    if model == "lgbm":
        return LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=8,
        )
    return RandomForestRegressor(
        n_estimators=500, random_state=seed, n_jobs=8, max_depth=None
    )


def train_and_eval(df: pd.DataFrame, seed: int, model: str, train_frac: float) -> dict:
    feature_cols = [c for c in df.columns if c not in {"id", "label"}]
    X = df[feature_cols].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_frac, random_state=seed
    )
    reg = build_model(model, seed)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "r2": float(r2_score(y_test, preds)),
        "n_features": int(X.shape[1]),
        "n_samples": int(len(df)),
        "model": model,
        "seed": seed,
        "train_frac": train_frac,
    }


def summarize_runs(runs: list[dict]) -> dict:
    metrics = {"mae": [], "rmse": [], "r2": []}
    for run in runs:
        for key in metrics:
            metrics[key].append(run[key])
    return {
        "mae_mean": float(np.mean(metrics["mae"])),
        "mae_std": float(np.std(metrics["mae"])),
        "rmse_mean": float(np.mean(metrics["rmse"])),
        "rmse_std": float(np.std(metrics["rmse"])),
        "r2_mean": float(np.mean(metrics["r2"])),
        "r2_std": float(np.std(metrics["r2"])),
        "n_runs": len(runs),
    }


def run_automl(
    df: pd.DataFrame,
    seed: int,
    n_iter: int,
    cv: int,
    n_jobs: int,
    models: list[str],
    train_frac: float,
) -> dict:
    feature_cols = [c for c in df.columns if c not in {"id", "label"}]
    X = df[feature_cols].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_frac, random_state=seed
    )

    results = {}

    if "xgb" in models:
        xgb = XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
        )
        xgb_params = {
            "n_estimators": [300, 600, 900],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 5, 10],
        }
        xgb_search = RandomizedSearchCV(
            xgb,
            xgb_params,
            n_iter=n_iter,
            cv=cv,
            random_state=seed,
            n_jobs=n_jobs,
            scoring="neg_mean_absolute_error",
        )
        xgb_search.fit(X_train, y_train)
        xgb_best = xgb_search.best_estimator_
        xgb_preds = xgb_best.predict(X_test)
        results["xgb_best"] = {
            "mae": float(mean_absolute_error(y_test, xgb_preds)),
            "rmse": float(mean_squared_error(y_test, xgb_preds) ** 0.5),
            "r2": float(r2_score(y_test, xgb_preds)),
            "params": xgb_search.best_params_,
        }

    if "lgbm" in models:
        lgbm = LGBMRegressor(
            random_state=seed,
            n_jobs=n_jobs,
        )
        lgbm_params = {
            "n_estimators": [300, 600, 900],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "num_leaves": [31, 63, 127, 255],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_samples": [5, 10, 20],
        }
        lgbm_search = RandomizedSearchCV(
            lgbm,
            lgbm_params,
            n_iter=n_iter,
            cv=cv,
            random_state=seed,
            n_jobs=n_jobs,
            scoring="neg_mean_absolute_error",
        )
        lgbm_search.fit(X_train, y_train)
        lgbm_best = lgbm_search.best_estimator_
        lgbm_preds = lgbm_best.predict(X_test)
        results["lgbm_best"] = {
            "mae": float(mean_absolute_error(y_test, lgbm_preds)),
            "rmse": float(mean_squared_error(y_test, lgbm_preds) ** 0.5),
            "r2": float(r2_score(y_test, lgbm_preds)),
            "params": lgbm_search.best_params_,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train regressor on contrastive embeddings.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "soap_pipeline_clean/outputs/set_transformer/contrastive/embeddings.csv",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "comb_id_labels.csv",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--models", nargs="+", default=["rf", "ridge", "gbr", "svr", "xgb", "lgbm"])
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--automl", action="store_true")
    parser.add_argument("--automl-iter", type=int, default=15)
    parser.add_argument("--automl-cv", type=int, default=3)
    parser.add_argument("--automl-jobs", type=int, default=4)
    parser.add_argument("--automl-models", nargs="+", default=["xgb", "lgbm"])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "soap_pipeline_clean/outputs/set_transformer/contrastive/regression_metrics.json",
    )
    args = parser.parse_args()

    embeddings = load_embeddings(args.embeddings)
    labels = load_labels(args.labels)
    dataset = build_dataset(embeddings, labels)

    results = {"runs": {}, "summary": {}}
    for model in args.models:
        runs = []
        for seed in args.seeds:
            metrics = train_and_eval(dataset, seed, model, args.train_frac)
            runs.append(metrics)
            print(json.dumps(metrics, indent=2))
        results["runs"][model] = runs
        results["summary"][model] = summarize_runs(runs)

    if args.automl:
        results["automl"] = run_automl(
            dataset,
            args.seeds[0],
            args.automl_iter,
            args.automl_cv,
            args.automl_jobs,
            args.automl_models,
            args.train_frac,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
