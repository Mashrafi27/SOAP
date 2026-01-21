#!/usr/bin/env python3
"""Export embeddings using a pretrained SetTransformer encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from soap_pipeline_clean.set_transformer.models import SetTransformerEncoder


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["filename"] = df["filename"].str.strip()
    return df


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
    parser = argparse.ArgumentParser(description="Export embeddings for a manifest using a pretrained encoder.")
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--soap-dir", type=Path, default=ROOT / "soap_pipeline_clean/outputs/soap_2d")
    parser.add_argument("--soap-meta", type=Path, default=ROOT / "soap_pipeline_clean/outputs/soap_2d/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=510)
    parser.add_argument("--num-inds", type=int, default=32)
    parser.add_argument("--dim-hidden", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads(args.soap_meta.read_text())
    feature_cols = meta["feature_labels"]
    max_atoms = meta["max_atoms"]

    encoder = SetTransformerEncoder(
        feature_dim=len(feature_cols),
        latent_dim=args.latent_dim,
        num_inds=args.num_inds,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
    ).to(device)

    state = torch.load(args.encoder, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    encoder.load_state_dict(state, strict=True)

    manifest = load_manifest(args.manifest)
    export_embeddings(
        encoder,
        manifest,
        args.soap_dir,
        feature_cols,
        max_atoms,
        args.output,
        device,
    )


if __name__ == "__main__":
    main()
