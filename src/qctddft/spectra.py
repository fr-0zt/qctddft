from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

# --- Gaussian lineshape (FWHM=sigma) ---
def gaussian(x, e_i, f_wi, sigma_i):
    factor = 2 * np.sqrt(np.log(2) / np.pi) * (f_wi / sigma_i)
    exponent = -((2 * (x - e_i) * np.sqrt(np.log(2)) / sigma_i) ** 2)
    return factor * np.exp(exponent)

def build_spectrum(energies: np.ndarray, strengths: np.ndarray, sigmas: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for e, f, s in zip(energies, strengths, sigmas):
        if f <= 0.0:
            continue
        y += gaussian(x, e, f, s)
    return y

def build_normalized_spectrum(
    df: pd.DataFrame,
    energy_cols: list[str],
    strength_cols: list[str],
    x: np.ndarray,
    sigma: float = 0.04,
    qualify_f1: float = 0.1,
    fmin_state: float = 0.0,
) -> tuple[np.ndarray, int, int]:
    """
    Normalize by count of qualified snapshots: (E1>0 & f1 >= qualify_f1).
    Then include all states with strength >= fmin_state.
    Returns (y_norm, n_qualified_snapshots, n_states_used).
    """
    E = df[energy_cols].to_numpy(float)
    F = df[strength_cols].to_numpy(float)

    qual = (E[:, 0] > 0.0) & (F[:, 0] >= qualify_f1)
    n_qual = int(qual.sum())

    mask_states = F >= fmin_state
    E_use = E[qual][mask_states[qual]]
    F_use = F[qual][mask_states[qual]]

    energies = E_use.ravel()
    strengths = F_use.ravel()
    sigmas = np.full_like(energies, sigma)
    y = build_spectrum(energies, strengths, sigmas, x)

    if n_qual > 0:
        y /= float(n_qual)
    return y, n_qual, int(energies.size)

