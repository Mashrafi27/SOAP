#!/usr/bin/env python3
"""
Train a SetTransformer regressor directly on the 2D SOAP matrices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import trange

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from soap_pipeline_clean.set_transformer.data import (
    SoapSetDataset,
    build_loader,
    stratified_split_dataset,
    SplitConfig,
)
from soap_pipeline_clean.set_transformer.models import SetTransformerRegressor


def load_labels(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["id"] = df["id"].str.strip()
    return df.set_index("id")["label"]


def load_manifest(path: Path, use_original_only: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    if use_original_only:
        df = df[df["origin"] == "original"]
    return df.reset_index(drop=True)


def evaluate(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds_list = []
    labels_list = []
    with torch.no_grad():
        for batch in loader:
            x, lengths, labels = batch
            x = x.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            preds, _ = model(x, lengths)
            preds_list.append(preds.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    y_true = np.concatenate(labels_list)
    y_pred = np.concatenate(preds_list)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.manifest, use_original_only=args.original_only)
    labels = load_labels(args.labels)
    meta = json.loads(args.soap_meta.read_text())
    feature_cols = meta["feature_labels"]
    max_atoms = meta["max_atoms"]

    dataset = SoapSetDataset(
        manifest_df=manifest,
        labels=labels,
        soap_dir=args.soap_dir,
        feature_columns=feature_cols,
        max_atoms=max_atoms,
    )

    split_cfg = SplitConfig(args.train_frac, args.val_frac, args.test_frac)
    train_ds, val_ds, test_ds = stratified_split_dataset(dataset, split_cfg, seed=args.seed)

    train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = build_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = build_loader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = SetTransformerRegressor(
        feature_dim=len(feature_cols),
        latent_dim=args.latent_dim,
        num_inds=args.num_inds,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        predictor_hidden=args.predictor_hidden,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None

    for epoch in trange(1, args.epochs + 1, desc="Training"):
        model.train()
        for batch in train_loader:
            x, lengths, labels_batch = batch
            x = x.to(device)
            lengths = lengths.to(device)
            labels_batch = labels_batch.to(device)

            preds, _ = model(x, lengths)
            loss = criterion(preds, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if best_state:
        torch.save(best_state, output_dir / "set_transformer_regressor.pt")

    model.load_state_dict(best_state["model"])
    test_metrics = evaluate(model, test_loader, device)
    summary = {
        "val_metrics": best_state["val_metrics"],
        "test_metrics": test_metrics,
        "config": vars(args),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("Best validation metrics:", best_state["val_metrics"])
    print("Test metrics:", test_metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SetTransformer on SOAP matrices.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "soap_pipeline_clean/metadata/mof_manifest.csv")
    parser.add_argument("--soap-dir", type=Path, default=ROOT / "soap_pipeline_clean/outputs/soap_2d")
    parser.add_argument("--soap-meta", type=Path, default=ROOT / "soap_pipeline_clean/outputs/soap_2d/manifest.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "comb_id_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "soap_pipeline_clean/outputs/set_transformer/regression")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-inds", type=int, default=32)
    parser.add_argument("--dim-hidden", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--predictor-hidden", type=int, default=256)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--original-only", action="store_true", help="Train only on the original 3k subset.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
