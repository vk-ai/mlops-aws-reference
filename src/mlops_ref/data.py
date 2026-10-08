"""Statlog (German Credit Data), numeric version. UCI ML Repository, CC BY 4.0.

1,000 applicants, 24 numeric attributes, label 1 = good / 2 = bad credit risk.
The file is small (102 KB) so it is bundled in ``data/raw``; ``fetch`` re-downloads
it from UCI and refuses it if the SHA-256 doesn't match.

Column names: the numeric file's documentation doesn't name its columns, so they
are ``a01``..``a24`` except three verified against the original categorical
file: a02 = duration in months, a04 = credit amount (hundreds of DM, rounded),
a10 = age in years.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
SHA256 = "2752b044394958ab6dd193a0b56ca0f0b3a2d8bc7cb8c008e35a5e84bbec02f8"
RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "german.data-numeric"

NAMED = {2: "duration_months", 4: "credit_amount", 10: "age_years"}
FEATURES = [NAMED.get(i, f"a{i:02d}") for i in range(1, 25)]
TARGET = "default"


class ChecksumError(ValueError):
    pass


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path: Path, expected: str = SHA256) -> str:
    got = sha256_of(path)
    if got != expected:
        raise ChecksumError(f"{path}: sha256 {got} != expected {expected}")
    return got


def fetch(dest: Path = RAW_PATH, url: str = URL, expected: str = SHA256) -> Path:
    """Download to a temp file, verify, then move into place."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - fixed public URL
        tmp.write_bytes(r.read())
    try:
        verify(tmp, expected)
    except ChecksumError:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)
    return dest


def load_raw(path: Path = RAW_PATH, expected: str | None = SHA256) -> pd.DataFrame:
    if expected:
        verify(path, expected)
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != 25:
        raise ValueError(f"expected 25 columns, got {df.shape[1]}")
    df.columns = FEATURES + ["label"]
    df[TARGET] = (df.pop("label") == 2).astype("int64")
    return df.astype({c: "float64" for c in FEATURES})
