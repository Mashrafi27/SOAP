#!/usr/bin/env python3
"""
Contrastive pretraining for SetTransformer encoders over 2D SOAP matrices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from soap_pipeline_clean.set_transformer.data import SoapSetDataset, build_loader
from soap_pipeline_clean.set_transformer.models import SetTransformerEncoder

META_COLUMNS = ["atom_index", "element", "x_ang", "y_ang", "z_ang"]


def load_labels(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["id"] = df["id"].str.strip()
    return df.set_index("id")["label"]


def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


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


def simclr_loss(z1, z2, temperature: float):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets])
    loss = F.cross_entropy(sim, targets)
    return loss


def pretrain(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.manifest)
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

    loader = build_loader(dataset, batch_size=args.batch_size, shuffle=True)

    encoder = SetTransformerEncoder(
        feature_dim=len(feature_cols),
        latent_dim=args.latent_dim,
        num_inds=args.num_inds,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
    ).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in trange(1, args.epochs + 1, desc="Contrastive pretraining"):
        encoder.train()
        epoch_loss = 0.0
        for batch in loader:
            x, lengths, _ = batch
            x = x.to(device)
            lengths = lengths.to(device)

            view1, len1 = augment_batch(x, lengths, args.atom_drop, args.feature_drop)
            view2, len2 = augment_batch(x, lengths, args.atom_drop, args.feature_drop)

            z1 = encoder(view1, len1)
            z2 = encoder(view2, len2)
            loss = simclr_loss(z1, z2, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch} contrastive loss: {epoch_loss / len(loader):.4f}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": encoder.state_dict(), "config": vars(args)}, output_dir / "encoder.pt")

    export_embeddings(encoder, manifest, args.soap_dir, feature_cols, max_atoms, output_dir / "embeddings.csv", device)


def export_embeddings(
    encoder: SetTransformerEncoder,
    manifest: pd.DataFrame,
    soap_dir: Path,
    feature_cols,
    max_atoms: int,
    output_path: Path,
    device: torch.device,
) -> None:
    encoder.eval()
    rows = []
    ids = []
    with torch.no_grad():
        for stem in manifest["filename"]:
            csv_path = soap_dir / f"{stem}.csv"
            df = pd.read_csv(csv_path, usecols=feature_cols)
            feats = torch.from_numpy(df.to_numpy(dtype=np.float32)).to(device)
            length = feats.shape[0]
            feats = feats.unsqueeze(0)
            pad = torch.zeros(1, max_atoms, feats.size(-1), device=device)
            pad[0, :length] = feats[0]
            lengths = torch.tensor([length], device=device)
            emb = encoder(pad, lengths).squeeze(0).cpu().numpy()
            rows.append(emb)
            ids.append(stem)

    columns = [f"embed_{i}" for i in range(rows[0].shape[0])]
    df = pd.DataFrame(rows, columns=columns)
    df.insert(0, "id", ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved embeddings to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrastive pretraining for SetTransformer.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "soap_pipeline_clean/metadata/mof_manifest.csv")
    parser.add_argument("--soap-dir", type=Path, default=ROOT / "soap_pipeline_clean/outputs/soap_2d")
    parser.add_argument("--soap-meta", type=Path, default=ROOT / "soap_pipeline_clean/outputs/soap_2d/manifest.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "comb_id_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "soap_pipeline_clean/outputs/set_transformer/contrastive")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=510)
    parser.add_argument("--num-inds", type=int, default=32)
    parser.add_argument("--dim-hidden", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--atom-drop", type=float, default=0.2)
    parser.add_argument("--feature-drop", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    pretrain(parse_args())
