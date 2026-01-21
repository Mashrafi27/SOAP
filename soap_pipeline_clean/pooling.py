"""
Utilities for aggregating per-atom SOAP descriptors into fixed-length vectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

META_COLUMNS = ["atom_index", "element", "x_ang", "y_ang", "z_ang"]


def load_feature_matrix(csv_path: Path, feature_columns: List[str]) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=META_COLUMNS + feature_columns)
    return df[feature_columns].to_numpy()


def inner_average(matrix: np.ndarray) -> np.ndarray:
    return matrix.mean(axis=0)


def max_pool(matrix: np.ndarray) -> np.ndarray:
    return matrix.max(axis=0)


def per_column_pca(matrix: np.ndarray) -> np.ndarray:
    """Replicate the historical PCA pooling (1 component per column)."""
    results: List[float] = []
    for col_idx in range(matrix.shape[1]):
        column = matrix[:, col_idx]
        if np.allclose(column, column[0]):
            results.append(0.0)
            continue
        model = PCA(n_components=1)
        transformed = model.fit_transform(column.reshape(-1, 1))
        results.append(float(transformed[0, 0]))
    return np.asarray(results, dtype=float)


POOLING_FUNCS = {
    "inner": inner_average,
    "max": max_pool,
    "pca": per_column_pca,
}
