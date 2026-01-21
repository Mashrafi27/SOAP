#!/usr/bin/env python3
"""Create a randomized test split and save ID lists and manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed train/test splits.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "soap_pipeline_clean/metadata/mof_manifest.csv")
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "soap_pipeline_clean/metadata/splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    df["filename"] = df["filename"].str.strip()
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    test_size = int(round(len(df) * args.test_frac))
    test_df = df.iloc[:test_size].copy()
    trainval_df = df.iloc[test_size:].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_ids.txt").write_text("\n".join(test_df["filename"].tolist()))
    (args.output_dir / "trainval_ids.txt").write_text("\n".join(trainval_df["filename"].tolist()))

    test_df.to_csv(args.output_dir / "test_manifest.csv", index=False)
    trainval_df.to_csv(args.output_dir / "trainval_manifest.csv", index=False)

    summary = {
        "seed": args.seed,
        "test_frac": args.test_frac,
        "n_total": len(df),
        "n_test": len(test_df),
        "n_trainval": len(trainval_df),
        "n_test_original": int((test_df["origin"] == "original").sum()),
        "n_test_new": int((test_df["origin"] == "new").sum()),
        "n_trainval_original": int((trainval_df["origin"] == "original").sum()),
        "n_trainval_new": int((trainval_df["origin"] == "new").sum()),
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
