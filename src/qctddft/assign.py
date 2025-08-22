from __future__ import annotations
import numpy as np
import pandas as pd
from math import erf, sqrt, log
from typing import Tuple, List
from .regions import Region

def _gauss_region_weight(E: np.ndarray, sigma: float, L: float, R: float) -> np.ndarray:
    # integral of normalized Gaussian over [L,R] (FWHM=sigma)
    alpha = 2.0*sqrt(log(2.0))/max(1e-12, sigma)
    return 0.5*(erf(alpha*(R - E)) - erf(alpha*(L - E)))

def assign_states_to_regions(
    df: pd.DataFrame,
    energy_cols: list[str],
    strength_cols: list[str],
    config_col: str,
    regions: List[Region],
    *,
    which_states: str = "first",  # "first" or "all"
    qualify_f1: float = 0.1,
    fmin_state: float = 0.0,
    sigma: float = 0.04,
    fractional: bool = True,
):
    """Return (assignment_df, summary_df) with configuration preserved."""
    E = df[energy_cols].to_numpy(float)
    F = df[strength_cols].to_numpy(float)
    cfg = df[config_col].to_numpy()

    qualify = (E[:, 0] > 0.0) & (F[:, 0] >= qualify_f1)
    if which_states == "first":
        E_use = E[qualify, [0]]
        F_use = F[qualify, [0]]
        cfg_use = cfg[qualify]
    else:
        E_use = E[qualify]
        F_use = F[qualify]
        cfg_use = np.repeat(cfg[qualify], E_use.shape[1])

    E_flat = E_use.ravel()
    F_flat = F_use.ravel()
    cfg_flat = cfg_use

    keep = F_flat >= fmin_state
    E_flat, F_flat, cfg_flat = E_flat[keep], F_flat[keep], cfg_flat[keep]

    L = np.array([r.left_energy for r in regions])
    R = np.array([r.right_energy for r in regions])
    J = len(regions)

    if fractional:
        W = np.vstack([_gauss_region_weight(E_flat, sigma, L[j], R[j]) for j in range(J)]).T
        row_sums = W.sum(axis=1)
        nz = row_sums > 0
        W[nz] /= row_sums[nz, None]
    else:
        W = np.zeros((E_flat.size, J))
        mids = 0.5*(L+R)
        for i, Ei in enumerate(E_flat):
            inside = (Ei >= L) & (Ei <= R)
            if inside.any():
                j = int(np.argmin(np.abs(mids[inside] - Ei)))
                idx = np.arange(J)[inside][j]
                W[i, idx] = 1.0

    # Build assignment table
    rows = []
    for i in range(E_flat.size):
        for j in range(J):
            if W[i, j] > 0.0:
                rows.append([int(cfg_flat[i]), float(E_flat[i]), float(F_flat[i]), j+1, float(W[i, j])])
    assignment_df = pd.DataFrame(rows, columns=["Configuration","energy","strength","region","weight"])

    # Summary per-region
    sums = []
    for j, r in enumerate(regions, 1):
        col = W[:, j-1]
        mask = col > 0
        eff_states = col.sum() if fractional else mask.sum()
        uniq_cfg = len(np.unique(cfg_flat[mask])) if mask.any() else 0
        f_sum = float((F_flat * col).sum())
        sums.append([j, r.left_energy, r.right_energy, r.peak_energy, r.peak_intensity,
                     float(eff_states) if fractional else int(eff_states),
                     int(uniq_cfg), f_sum])
    summary_df = pd.DataFrame(
        sums,
        columns=["region","left_e","right_e","peak_e","peak_intensity",
                 "effective_states" if fractional else "n_states",
                 "unique_snapshots","f_sum"]
    )
    return assignment_df, summary_df

