from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from . import logger
from .plots import plot_spectrum

def _gaussian(x, e_i, f_wi, sigma_i):
    """Calculates the Gaussian function for a single transition."""
    alpha = 2.0 * np.sqrt(np.log(2.0)) / sigma_i
    factor = 2.0 * np.sqrt(np.log(2.0) / np.pi) * (f_wi / sigma_i)
    return factor * np.exp(-(alpha * (x - e_i)) ** 2)

def build_normalized_spectrum(
    extracted_tsv: str,
    sigma: float = 0.04,
    emin: float = 1.7,
    emax: float = 2.4,
    npts: int = 1000,
    fmin_snapshot: float = 0.10,
    fmin_state: float = 0.0,
    states: str = "all",
    save_plot: bool = False,
    output_path: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds a normalized, broadened spectrum from the extracted TDDFT data.

    Args:
        extracted_tsv: Path to the tab-separated file with the extracted data.
        sigma: The broadening factor (FWHM) for the Gaussian functions.
        emin: The minimum energy for the spectrum.
        emax: The maximum energy for the spectrum.
        npts: The number of points in the spectrum.
        fmin_snapshot: The oscillator strength threshold for the first state.
        fmin_state: The oscillator strength threshold for individual states.
        states: Whether to use 'all' states or only the 'first'.
        save_plot: If True, generates and saves a plot of the spectrum.
        output_path: The base path for the output files.

    Returns:
        A tuple containing the energy and intensity arrays of the spectrum.
    """
    df = pd.read_csv(extracted_csv, sep=",")
    energy_cols = [c for c in df.columns if c.startswith("Energy ")]
    strength_cols = [c for c in df.columns if c.startswith("Strength ")]
    if not energy_cols or not strength_cols:
        raise ValueError("Input must have Energy*/Strength* columns (tab-separated).")

    energy_cols.sort(key=lambda s: int(s.split()[-1]))
    strength_cols.sort(key=lambda s: int(s.split()[-1]))

    E = df[energy_cols].to_numpy(float)
    F = df[strength_cols].to_numpy(float)

    mask = (E[:, 0] > 0.0) & (F[:, 0] >= fmin_snapshot)
    if not np.any(mask):
        raise RuntimeError("No snapshots qualify under the first-state rule.")
    E = E[mask]; F = F[mask]
    n_qual = int(mask.sum())
    logger.info("Qualifying snapshots: %d", n_qual)

    if states == "first":
        e = E[:, 0]
        f = F[:, 0]
    else:
        e = E.ravel()
        f = F.ravel()

    if fmin_state > 0.0:
        keep = f >= fmin_state
        e, f = e[keep], f[keep]

    keep = (e > 0.0) & np.isfinite(e) & np.isfinite(f)
    e, f = e[keep], f[keep]

    x = np.linspace(emin, emax, npts)
    spec = np.zeros_like(x, dtype=float)
    sig = np.full_like(e, sigma, dtype=float)
    for ei, fi, si in zip(e, f, sig):
        spec += _gaussian(x, ei, fi, si)
    spec /= float(n_qual)
    
    if save_plot and output_path:
        plot_path = output_path.replace(".csv", ".png")
        plot_spectrum(x, spec, save_path=plot_path)

    return x, spec

def write_spectrum_csv(out_csv: str, x: np.ndarray, y: np.ndarray) -> None:
    pd.DataFrame({"Energy (eV)": x, "Intensity": y}).to_csv(out_csv, index=False)
