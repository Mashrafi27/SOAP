from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


@dataclass
class SplitConfig:
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1

    def validate(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")


class SoapSetDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        labels: pd.Series,
        soap_dir: Path,
        feature_columns: List[str],
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


def stratified_split_dataset(
    dataset: SoapSetDataset,
    split_cfg: SplitConfig,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    split_cfg.validate()
    total = len(dataset)
    train_len = int(total * split_cfg.train_frac)
    val_len = int(total * split_cfg.val_frac)
    test_len = total - train_len - val_len
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=generator)


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
