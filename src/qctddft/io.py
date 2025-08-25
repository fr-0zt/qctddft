from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from . import logger
from .regions import Region # Import the Region class

# -------------------------
# Generic readers/writers
# -------------------------

def write_regions_csv(out_csv: str, regions: List[Region]) -> None:
    """Saves identified spectral regions to a CSV file."""
    if not regions:
        logger.warning("No regions were provided to save.")
        return
    
    # Convert list of dataclasses to a list of dicts for DataFrame creation
    region_data = [
        {
            "region_id": i + 1,
            "left_energy_ev": r.left_energy,
            "right_energy_ev": r.right_energy,
            "peak_energy_ev": r.peak_energy,
            "peak_intensity": r.peak_intensity,
            "width_ev": r.right_energy - r.left_energy,
        }
        for i, r in enumerate(regions)
    ]
    
    df = pd.DataFrame(region_data)
    df.to_csv(out_csv, index=False, float_format="%.5f")
    logger.info(f"Saved {len(regions)} regions to {out_csv}")

def read_regions_csv(path: str) -> List[Region]:
    """Reads a regions CSV file and reconstructs a list of Region objects."""
    df = pd.read_csv(path)
    regions = []
    # The indices (left_idx, etc.) aren't needed for the assignment logic,
    # so we can use placeholder values (0).
    for row in df.itertuples():
        regions.append(
            Region(
                left_idx=0,
                right_idx=0,
                left_energy=row.left_energy_ev,
                right_energy=row.right_energy_ev,
                peak_idx=0,
                peak_energy=row.peak_energy_ev,
                peak_intensity=row.peak_intensity,
            )
        )
    return regions
    
def read_spectrum_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read spectrum CSV with columns like 'Energy (eV)', 'Intensity' (case-insensitive)."""
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    # Try flexible matching
    e_col = _first_match(cols, ["energy (ev)", "energy_ev", "energy", "e"])
    i_col = _first_match(cols, ["intensity", "i", "y"])
    if e_col is None or i_col is None:
        raise ValueError("Spectrum CSV must contain Energy and Intensity columns.")
    x = df[df.columns[e_col]].to_numpy(dtype=float)
    y = df[df.columns[i_col]].to_numpy(dtype=float)
    if not np.all(np.diff(x) > 0):
        raise ValueError("Energies must be strictly increasing.")
    return x, y


def read_tddft_table(path: str) -> Tuple[pd.DataFrame, List[str], List[str], str]:
    """
    Read an extracted TDDFT table (.csv or .dat).
    Expected columns:
      - 'Configuration' (or 'config', 'cfg')
      - 'Energy i' and 'Strength i' (i = 1..N)
    Returns: (df, energy_cols, strength_cols, config_col)
    """
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        # try tab-delimited
        df = pd.read_csv(p, sep=r"\s+|\t", engine="python")
    df_cols = [c.strip() for c in df.columns]
    lower = [c.lower() for c in df_cols]

    cfg_idx = _first_match(lower, ["configuration", "config", "cfg"])
    if cfg_idx is None:
        raise ValueError("Missing 'Configuration' column in extracted table.")
    config_col = df_cols[cfg_idx]

    # Energy/Strength columns (ordered by index)
    e_cols = []
    f_cols = []
    for c in df_cols:
        cl = c.lower()
        mE = re.match(r"energy\s*(\d+)", cl)
        mF = re.match(r"(?:strength|oscillator[_\s]*strength)\s*(\d+)", cl)
        if mE:
            e_cols.append(c)
        if mF:
            f_cols.append(c)
    # stable sort by trailing number
    def _key(c: str) -> int:
        m = re.search(r"(\d+)\s*$", c)
        return int(m.group(1)) if m else 0

    e_cols.sort(key=_key)
    f_cols.sort(key=_key)
    if not e_cols or not f_cols:
        raise ValueError("No Energy/Strength columns found (expected 'Energy i' / 'Strength i').")

    return df, e_cols, f_cols, config_col


def write_meta_sidecar(csv_path: str, meta: dict) -> None:
    """Write <csv>.meta.json next to CSV for reproducibility."""
    meta_path = Path(csv_path).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Saved metadata → {meta_path.name}")


def _first_match(names: List[str], candidates: List[str]) -> int | None:
    for target in candidates:
        if target in names:
            return names.index(target)
    return None