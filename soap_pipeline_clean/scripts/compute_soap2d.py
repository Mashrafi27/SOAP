from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from soap_pipeline_clean.generator import Soap2DGenerator, SoapConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate per-atom (2D) SOAP descriptors for a CIF dataset.",
    )
    parser.add_argument("--cif-dir", type=Path, required=True, help="Directory containing .cif files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store descriptor files.")
    parser.add_argument("--r-cut", type=float, default=5.0, help="Cutoff radius (Å).")
    parser.add_argument("--n-max", type=int, default=1, help="Number of radial basis functions.")
    parser.add_argument("--l-max", type=int, default=1, help="Max angular momentum.")
    parser.add_argument("--sigma", type=float, default=None, help="Gaussian width (Å).")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for SOAP creation (-1 uses all cores).")
    parser.add_argument(
        "--periodic",
        action="store_true",
        help="Treat structures as periodic when building the SOAP descriptor.",
    )
    parser.add_argument(
        "--file-format",
        choices=("csv", "parquet"),
        default="csv",
        help="Output format for descriptor tables.",
    )
    parser.add_argument(
        "--no-positions",
        action="store_true",
        help="Do not include Cartesian coordinates in the output tables.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="manifest.json",
        help="Name of the JSON manifest written alongside outputs.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Raise an error instead of overwriting existing descriptor files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = SoapConfig(
        cif_dir=args.cif_dir,
        output_dir=args.output_dir,
        r_cut=args.r_cut,
        n_max=args.n_max,
        l_max=args.l_max,
        sigma=args.sigma,
        periodic=args.periodic,
        n_jobs=args.n_jobs,
        file_format=args.file_format,
        include_positions=not args.no_positions,
        overwrite=not args.no_overwrite,
        manifest_name=args.manifest_name,
    )

    generator = Soap2DGenerator(config)
    generator.run()


if __name__ == "__main__":
    main()
