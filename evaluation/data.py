"""Load the modelling dataset as an aligned (target, exogenous) pair."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TARGET_COL = "brent"
EXOG_COLS = [
    "dollar_return",
    "indpro_return",
    "inventory_pct",
    "fed_funds_diff",
    "vix_diff",
]

WEEKLY_FILE = "oil_macro_weekly.csv"
TRANSFORMED_FILE = "oil_macro_transformed.csv"


def load_dataset(data_dir: Path = DATA_DIR) -> tuple[pd.Series, pd.DataFrame]:
    """Return the Brent price series and the exogenous matrix on a shared index.

    Mirrors the alignment in ``train.py`` so the backtest evaluates exactly
    the model the app serves.
    """
    weekly = pd.read_csv(data_dir / WEEKLY_FILE, index_col=0, parse_dates=True)
    transformed = pd.read_csv(data_dir / TRANSFORMED_FILE, index_col=0, parse_dates=True)

    common = weekly.index.intersection(transformed.index)
    y = weekly.loc[common, TARGET_COL].astype(float)
    X = transformed.loc[common, EXOG_COLS].astype(float)

    if y.isna().any() or X.isna().any().any():
        raise ValueError("Dataset contains missing values after alignment")

    freq = pd.infer_freq(y.index)
    if freq is None:
        raise ValueError("Index is not a regular weekly series; gaps would break walk-forward slicing")
    y.index.freq = freq
    X.index.freq = freq
    return y, X


def train_size(n_obs: int, train_frac: float = 0.8) -> int:
    """Number of training observations, using the same rule as ``train.py``."""
    return int(n_obs * train_frac)


def content_hash(path: Path) -> str:
    """SHA-256 of a text file with line endings normalised to LF.

    Git on Windows may check files out with CRLF line endings; hashing raw
    bytes would then give different fingerprints for identical data on
    different machines.
    """
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()[:16]


def data_fingerprint(data_dir: Path = DATA_DIR) -> dict[str, str]:
    """Content hash of each input file, recorded with every backtest run."""
    return {name: content_hash(data_dir / name) for name in (WEEKLY_FILE, TRANSFORMED_FILE)}
