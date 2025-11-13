#!/usr/bin/env python3
"""
Generate a manifest describing which CIF entries belong to the original 3k
dataset versus the newer 3k, and provide helper metadata for batching.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]


def list_cif_stems(directory: Path) -> List[str]:
    return sorted(p.stem for p in directory.glob("*.cif"))


def write_manifest(
    comb_dir: Path,
    base_dir: Path,
    output_csv: Path,
) -> None:
    comb = list_cif_stems(comb_dir)
    base = set(list_cif_stems(base_dir))
    new_entries = sorted(stem for stem in comb if stem not in base)
    new_index_lookup = {stem: idx + 1 for idx, stem in enumerate(new_entries)}

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "filename",
                "origin",
                "combined_rank",
                "new_index",
                "new_batch_500",
            ]
        )

        for rank, stem in enumerate(comb, 1):
            if stem in base:
                origin = "original"
                new_index = ""
                new_batch = ""
            else:
                origin = "new"
                new_index = new_index_lookup[stem]
                new_batch = (new_index - 1) // 500 + 1

            writer.writerow([stem, origin, rank, new_index, new_batch])

    print(
        f"Wrote manifest for {len(comb)} MOFs "
        f"({len(base)} original, {len(new_entries)} new) -> {output_csv}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MOF manifest metadata.")
    parser.add_argument(
        "--comb-dir",
        type=Path,
        default=ROOT / "comb_CIF_files",
        help="Directory containing the combined CIF files (6089).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=ROOT / "CIF_files",
        help="Directory containing the original CIF files (≈3089).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "soap_pipeline_clean/metadata/mof_manifest.csv",
        help="Where to write the manifest CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_manifest(args.comb_dir, args.base_dir, args.output)


if __name__ == "__main__":
    main()
