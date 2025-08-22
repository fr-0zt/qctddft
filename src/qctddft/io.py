from __future__ import annotations
import os
from typing import Tuple, List
import numpy as np
import pandas as pd

CONFIG_CANDIDATES = ("Configuration", "Config", "Snapshot", "Frame")

def read_tddft_table(path: str) -> Tuple[pd.DataFrame, List[str], List[str], str]:
    """Read a QChem TDDFT extracted table (.dat/.csv), detect state columns and configuration column."""
    sep = "\t" if path.lower().endswith(".dat") else ","
    df = pd.read_csv(path, sep=sep)
    ecols = [c for c in df.columns if "Energy" in c]
    fcols = [c for c in df.columns if "Strength" in c]
    if not ecols or not fcols:
        raise ValueError("Could not detect Energy*/Strength* columns in TDDFT table.")

    cfg_col = None
    for cand in CONFIG_CANDIDATES:
        if cand in df.columns:
            cfg_col = cand
            break
    if cfg_col is None:
        raise ValueError(f"No configuration column found (tried {CONFIG_CANDIDATES}).")

    return df, ecols, fcols, cfg_col

def read_spectrum_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read spectrum CSV; accepts either canonical headers or first two columns."""
    df = pd.read_csv(path)
    if "Energy (eV)" in df.columns and "Intensity" in df.columns:
        x = df["Energy (eV)"].to_numpy(float)
        y = df["Intensity"].to_numpy(float)
    else:
        x = df.iloc[:, 0].to_numpy(float)
        y = df.iloc[:, 1].to_numpy(float)
    return x, y

