"""
High-level helpers for generating 2D SOAP descriptors in an organised manner.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ase.io import read
from dscribe.descriptors import SOAP
from tqdm import tqdm

from .columns import feature_labels, normalise_species


@dataclass
class SoapConfig:
    """Configuration bundle for 2D SOAP generation."""

    cif_dir: Path
    output_dir: Path
    r_cut: float
    n_max: int
    l_max: int
    sigma: Optional[float] = None
    periodic: bool = False
    n_jobs: int = 1
    file_format: str = "csv"
    overwrite: bool = True
    include_positions: bool = True
    include_species_counts: bool = False
    manifest_name: str = "manifest.json"

    def __post_init__(self) -> None:
        self.cif_dir = Path(self.cif_dir)
        self.output_dir = Path(self.output_dir)
        allowed_formats = {"csv", "parquet"}
        if self.file_format not in allowed_formats:
            raise ValueError(f"file_format must be one of {allowed_formats}")


class Soap2DGenerator:
    """
    Orchestrates discovery, descriptor construction, and persistence of 2D SOAP
    tensors (no averaging) for a set of CIF structures.
    """

    def __init__(self, config: SoapConfig) -> None:
        self.config = config
        self.cif_files = self._discover_cif_files()
        if not self.cif_files:
            raise FileNotFoundError(f"No CIF files found in {self.config.cif_dir}")

        self.all_columns: List[str] = []
        self._manifest_entries: List[Dict] = []
        self._errors: List[Dict] = []
        self._max_atoms: int = 0

    # ------------------------------------------------------------------ #
    # Preparation
    # ------------------------------------------------------------------ #
    def prepare(self) -> None:
        """Determine global feature columns by scanning every MOF."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_columns, self._max_atoms = self._collect_all_columns()

    def _discover_cif_files(self) -> List[Path]:
        return sorted(self.config.cif_dir.glob("*.cif"))

    def _collect_all_columns(self) -> Tuple[List[str], int]:
        """Scan every CIF to determine the union of actual SOAP columns."""
        all_cols: set[str] = set()
        max_atoms = 0
        for cif_path in tqdm(self.cif_files, desc="Scanning columns", leave=False):
            structure = read(cif_path)
            max_atoms = max(max_atoms, len(structure))
            species = normalise_species(structure.get_chemical_symbols())
            soap = self._build_descriptor(species)
            cols = feature_labels(soap, species)
            all_cols.update(cols)
        ordered_cols = sorted(all_cols)
        return ordered_cols, max_atoms

    def _build_descriptor(self, species: Sequence[str]) -> SOAP:
        soap_kwargs = dict(
            species=species,
            periodic=self.config.periodic,
            r_cut=self.config.r_cut,
            n_max=self.config.n_max,
            l_max=self.config.l_max,
            average="off",
            sparse=False,
        )
        if self.config.sigma is not None:
            soap_kwargs["sigma"] = self.config.sigma

        return SOAP(**soap_kwargs)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        if not self.all_columns:
            self.prepare()

        for cif_path in tqdm(self.cif_files, desc="Generating SOAP (2D)"):
            try:
                structure = read(cif_path)
                descriptor, per_mof_columns = self._compute_descriptor(structure)
                df = self._build_dataframe(structure, descriptor, per_mof_columns)
                output_path = self._write_dataframe(cif_path, df)
                self._manifest_entries.append(
                    {
                        "source": str(cif_path),
                        "output": str(output_path),
                        "n_atoms": len(structure),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                self._errors.append({"source": str(cif_path), "error": str(exc)})

        self._write_manifest()

    def _compute_descriptor(self, structure):
        species = normalise_species(structure.get_chemical_symbols())
        soap = self._build_descriptor(species)
        descriptor = soap.create(structure, n_jobs=self.config.n_jobs)
        per_mof_columns = feature_labels(soap, species)
        return descriptor, per_mof_columns

    def _build_dataframe(
        self, structure, descriptor: np.ndarray, per_mof_columns: Sequence[str]
    ) -> pd.DataFrame:
        df = pd.DataFrame(descriptor, columns=per_mof_columns)
        df = df.reindex(columns=self.all_columns, fill_value=0.0)

        # Prepend metadata columns
        df.insert(0, "atom_index", np.arange(len(structure)))
        df.insert(1, "element", structure.get_chemical_symbols())

        if self.config.include_positions:
            positions = structure.get_positions()
            df.insert(2, "x_ang", positions[:, 0])
            df.insert(3, "y_ang", positions[:, 1])
            df.insert(4, "z_ang", positions[:, 2])

        return df

    def _write_dataframe(self, cif_path: Path, df: pd.DataFrame) -> Path:
        filename = cif_path.stem
        output_path = self.config.output_dir / f"{filename}.{self.config.file_format}"

        if output_path.exists() and not self.config.overwrite:
            raise FileExistsError(f"{output_path} exists and overwrite=False")

        if self.config.file_format == "csv":
            df.to_csv(output_path, index=False)
        elif self.config.file_format == "parquet":
            df.to_parquet(output_path, index=False)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported file_format: {self.config.file_format}")

        return output_path

    # ------------------------------------------------------------------ #
    # Manifest helpers
    # ------------------------------------------------------------------ #
    def _write_manifest(self) -> None:
        manifest = {
            "config": self._manifest_config(),
            "n_features": len(self.all_columns),
            "feature_labels": self.all_columns,
            "max_atoms": self._max_atoms,
            "outputs": self._manifest_entries,
            "errors": self._errors,
        }
        manifest_path = self.config.output_dir / self.config.manifest_name
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    def _manifest_config(self) -> Dict:
        cfg_dict = asdict(self.config)
        cfg_dict["cif_dir"] = str(cfg_dict["cif_dir"])
        cfg_dict["output_dir"] = str(cfg_dict["output_dir"])
        return cfg_dict
