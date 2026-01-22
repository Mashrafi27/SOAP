#!/usr/bin/env python3
"""Contrastive pretraining with a gated-attention MIL encoder and pooled views."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from soap_pipeline_clean.set_transformer.mil_models import GatedAttentionEncoder
from soap_pipeline_clean.pooling import POOLING_FUNCS, load_feature_matrix


def read_id_list(path: Path | None) -> set[str]:
    if not path:
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["filename"] = df["filename"].str.strip()
    return df


def load_labels(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["id"] = df["id"].str.strip()
    return df.set_index("id")["label"]


def load_pooled_vectors(pool_dir: Path, methods: list[str], split: str) -> dict[str, dict[str, np.ndarray]]:
    pooled = {}
    for method in methods:
        path = pool_dir / f"{method}_{split}.csv"
        df = pd.read_csv(path)
        df["id"] = df["id"].str.strip()
        feature_cols = [c for c in df.columns if c not in {"id", "label"}]
        pooled[method] = {row["id"]: row[feature_cols].to_numpy(dtype=np.float32) for _, row in df.iterrows()}
    return pooled


def augment_batch(x, lengths, atom_drop: float, feature_drop: float):
    augmented = torch.zeros_like(x)
    new_lengths = torch.zeros_like(lengths)
    B, max_atoms, feat_dim = x.shape
    for i in range(B):
        length = max(1, int(lengths[i].item()))
        keep = max(1, int(length * (1 - atom_drop)))
        perm = torch.randperm(length, device=x.device)[:keep]
        view = x[i, :length][perm]
        if feature_drop > 0:
            mask = (torch.rand(feat_dim, device=x.device) > feature_drop).float()
            view = view * mask
        augmented[i, : view.size(0)] = view
        new_lengths[i] = view.size(0)
    return augmented, new_lengths


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z1 @ z2.T / temperature
    targets = torch.arange(z1.size(0), device=z1.device)
    loss1 = F.cross_entropy(logits, targets)
    loss2 = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss1 + loss2)


class MultiViewSoapDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        labels: pd.Series,
        soap_dir: Path,
        feature_columns: list[str],
        max_atoms: int,
        pooled: dict[str, dict[str, np.ndarray]],
    ):
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.labels = labels
        self.soap_dir = Path(soap_dir)
        self.feature_columns = feature_columns
        self.max_atoms = max_atoms
        self.feature_dim = len(feature_columns)
        self.pooled = pooled

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

        pooled_views = {m: self.pooled[m][stem] for m in self.pooled}
        label = float(self.labels[stem])

        return (
            torch.from_numpy(padded),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
            pooled_views,
        )


def collate_fn(batch):
    xs, lengths, labels, pooled_views = zip(*batch)
    x = torch.stack(xs)
    lengths = torch.stack(lengths)
    labels = torch.stack(labels)
    pooled = {key: torch.tensor([pv[key] for pv in pooled_views], dtype=torch.float32) for key in pooled_views[0]}
    return x, lengths, labels, pooled


def pretrain(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.wandb:
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    manifest = load_manifest(args.manifest)
    include_ids = read_id_list(args.include_ids)
    exclude_ids = read_id_list(args.exclude_ids)
    if include_ids:
        manifest = manifest[manifest["filename"].isin(include_ids)]
    if exclude_ids:
        manifest = manifest[~manifest["filename"].isin(exclude_ids)]
    manifest = manifest.reset_index(drop=True)

    labels = load_labels(args.labels)
    meta = json.loads(args.soap_meta.read_text())
    feature_cols = meta["feature_labels"]
    max_atoms = meta["max_atoms"]

    pooled = load_pooled_vectors(args.pooled_dir, args.pool_methods, args.pool_split)

    dataset = MultiViewSoapDataset(
        manifest_df=manifest,
        labels=labels,
        soap_dir=args.soap_dir,
        feature_columns=feature_cols,
        max_atoms=max_atoms,
        pooled=pooled,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    encoder = GatedAttentionEncoder(
        feature_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)

    proj = torch.nn.Linear(args.latent_dim, args.latent_dim).to(device)
    pool_heads = torch.nn.ModuleDict(
        {
            m: torch.nn.Linear(len(next(iter(pooled[m].values()))), args.latent_dim)
            for m in args.pool_methods
        }
    ).to(device)

    params = list(encoder.parameters()) + list(proj.parameters()) + list(pool_heads.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    for epoch in trange(1, args.epochs + 1, desc="Contrastive pretraining (MIL)"):
        encoder.train()
        epoch_loss = 0.0
        for batch in loader:
            x, lengths, _, pooled_views = batch
            x = x.to(device)
            lengths = lengths.to(device)

            if args.atom_drop or args.feature_drop:
                x_aug, len_aug = augment_batch(x, lengths, args.atom_drop, args.feature_drop)
            else:
                x_aug, len_aug = x, lengths

            z_enc = proj(encoder(x_aug, len_aug))
            z_enc = F.normalize(z_enc, dim=-1)

            loss = 0.0
            for method in args.pool_methods:
                pooled_vec = pooled_views[method].to(device)
                z_pool = pool_heads[method](pooled_vec)
                z_pool = F.normalize(z_pool, dim=-1)
                loss = loss + info_nce(z_enc, z_pool, args.temperature)
            loss = loss / len(args.pool_methods)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        if args.wandb:
            wandb.log({"epoch": epoch, "contrastive_loss": avg_loss})
        print(f"Epoch {epoch} loss: {avg_loss:.4f}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": encoder.state_dict(), "config": vars(args)}, output_dir / "encoder.pt")

    if args.wandb:
        wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrastive pretraining with MIL encoder + pooled views.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "soap_pipeline_clean/metadata/mof_manifest.csv")
    parser.add_argument("--soap-dir", type=Path, default=ROOT / "soap_pipeline_clean/outputs/soap_2d")
    parser.add_argument("--soap-meta", type=Path, default=ROOT / "soap_pipeline_clean/outputs/soap_2d/manifest.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "comb_id_labels.csv")
    parser.add_argument("--pooled-dir", type=Path, default=ROOT / "soap_pipeline_clean/outputs/pools")
    parser.add_argument("--pool-methods", nargs="+", default=["inner", "max", "pca"])
    parser.add_argument("--pool-split", type=str, default="trainval")
    parser.add_argument("--include-ids", type=Path, default=ROOT / "soap_pipeline_clean/metadata/splits/trainval_ids.txt")
    parser.add_argument("--exclude-ids", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "soap_pipeline_clean/outputs/set_transformer/contrastive_mil")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--atom-drop", type=float, default=0.2)
    parser.add_argument("--feature-drop", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="mof-settransformer")
    parser.add_argument("--wandb-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    pretrain(parse_args())
