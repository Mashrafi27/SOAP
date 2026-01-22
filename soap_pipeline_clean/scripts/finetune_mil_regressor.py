#!/usr/bin/env python3
"""Fine-tune a MIL regressor using a pretrained GatedAttention encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from soap_pipeline_clean.set_transformer.mil_models import GatedAttentionEncoder
from soap_pipeline_clean.set_transformer.data import SplitConfig, stratified_split_dataset


def load_labels(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["id"] = df["id"].str.strip()
    return df.set_index("id")["label"]


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["filename"] = df["filename"].str.strip()
    return df.reset_index(drop=True)


def read_id_list(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


class SoapMilDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        labels: pd.Series,
        soap_dir: Path,
        feature_columns: list[str],
        max_atoms: int,
    ):
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.labels = labels
        self.soap_dir = Path(soap_dir)
        self.feature_columns = feature_columns
        self.max_atoms = max_atoms
        self.feature_dim = len(feature_columns)

    def __len__(self) -> int:
        return len(self.manifest_df)

    def __getitem__(self, idx: int):
        row = self.manifest_df.iloc[idx]
        stem = row["filename"]
        csv_path = self.soap_dir / f"{stem}.csv"

        df = pd.read_csv(csv_path, usecols=self.feature_columns)
        feats = df.to_numpy(dtype=np.float32)
        length = feats.shape[0]

        padded = np.zeros((self.max_atoms, self.feature_dim), dtype=np.float32)
        padded[:length] = feats
        label = float(self.labels[stem])

        return torch.from_numpy(padded), torch.tensor(length, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)


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
            preds = model(x, lengths)
            preds_list.append(preds.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    y_true = np.concatenate(labels_list)
    y_pred = np.concatenate(preds_list)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def load_encoder_weights(encoder: GatedAttentionEncoder, encoder_ckpt: Path) -> None:
    state = torch.load(encoder_ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    encoder.load_state_dict(state, strict=True)


def set_encoder_requires_grad(encoder: GatedAttentionEncoder, requires_grad: bool) -> None:
    for param in encoder.parameters():
        param.requires_grad = requires_grad


class MilRegressor(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, latent_dim: int, predictor_hidden: int):
        super().__init__()
        self.encoder = GatedAttentionEncoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, predictor_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(predictor_hidden, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x, lengths)
        pred = self.head(emb).squeeze(-1)
        return pred


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.wandb:
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    manifest = load_manifest(args.manifest)
    labels = load_labels(args.labels)
    meta = json.loads(args.soap_meta.read_text())
    feature_cols = meta["feature_labels"]
    max_atoms = meta["max_atoms"]

    train_ids = read_id_list(args.train_ids)
    val_ids = read_id_list(args.val_ids)
    test_ids = read_id_list(args.test_ids)

    if train_ids or val_ids or test_ids:
        if not train_ids or not test_ids:
            raise ValueError("Both --train-ids and --test-ids are required for fixed splits.")

        train_manifest = manifest[manifest["filename"].isin(train_ids)].reset_index(drop=True)
        test_manifest = manifest[manifest["filename"].isin(test_ids)].reset_index(drop=True)

        if val_ids:
            val_manifest = manifest[manifest["filename"].isin(val_ids)].reset_index(drop=True)
            overlap = set(train_manifest["filename"]).intersection(val_manifest["filename"])
            if overlap:
                train_manifest = train_manifest[~train_manifest["filename"].isin(overlap)].reset_index(drop=True)
            train_ds = SoapMilDataset(train_manifest, labels, args.soap_dir, feature_cols, max_atoms)
            val_ds = SoapMilDataset(val_manifest, labels, args.soap_dir, feature_cols, max_atoms)
        else:
            train_all = SoapMilDataset(train_manifest, labels, args.soap_dir, feature_cols, max_atoms)
            split_cfg = SplitConfig(1 - args.val_frac, args.val_frac, 0.0)
            train_ds, val_ds, _ = stratified_split_dataset(train_all, split_cfg, seed=args.seed)

        test_ds = SoapMilDataset(test_manifest, labels, args.soap_dir, feature_cols, max_atoms)
    else:
        dataset = SoapMilDataset(manifest, labels, args.soap_dir, feature_cols, max_atoms)
        split_cfg = SplitConfig(args.train_frac, args.val_frac, args.test_frac)
        train_ds, val_ds, test_ds = stratified_split_dataset(dataset, split_cfg, seed=args.seed)

    train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = build_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = build_loader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MilRegressor(
        feature_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        predictor_hidden=args.predictor_hidden,
    ).to(device)

    if args.pretrained_encoder:
        load_encoder_weights(model.encoder, args.pretrained_encoder)

    if args.freeze_encoder:
        set_encoder_requires_grad(model.encoder, False)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None

    for epoch in trange(1, args.epochs + 1, desc="Training"):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            x, lengths, labels_batch = batch
            x = x.to(device)
            lengths = lengths.to(device)
            labels_batch = labels_batch.to(device)

            preds = model(x, lengths)
            loss = criterion(preds, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        if args.wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(epoch_losses)),
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_r2": val_metrics["r2"],
                }
            )
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
        torch.save(best_state, output_dir / "mil_regressor.pt")

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

    if args.wandb:
        wandb.log(
            {
                "best_val_mae": best_state["val_metrics"]["mae"],
                "best_val_rmse": best_state["val_metrics"]["rmse"],
                "best_val_r2": best_state["val_metrics"]["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )
        wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MIL regressor with pretrained encoder.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "soap_pipeline_clean/metadata/mof_manifest.csv")
    parser.add_argument("--soap-dir", type=Path, default=ROOT / "soap_pipeline_clean/outputs/soap_2d")
    parser.add_argument("--soap-meta", type=Path, default=ROOT / "soap_pipeline_clean/outputs/soap_2d/manifest.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "comb_id_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "soap_pipeline_clean/outputs/set_transformer/finetune_mil")
    parser.add_argument("--pretrained-encoder", type=Path, default=ROOT / "soap_pipeline_clean/outputs/set_transformer/contrastive_mil/encoder.pt")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--predictor-hidden", type=int, default=256)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ids", type=Path, default=None)
    parser.add_argument("--val-ids", type=Path, default=None)
    parser.add_argument("--test-ids", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="mof-settransformer")
    parser.add_argument("--wandb-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
