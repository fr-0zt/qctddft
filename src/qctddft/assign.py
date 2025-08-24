from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from .regions import Region
from . import logger

def _gauss_region_weight(Ei: np.ndarray, fwhm: float, L: float, R: float) -> np.ndarray:
    """
    Analytic fractional weight of each state at energy Ei within region [L, R]
    for a Gaussian with FWHM=fwhm. Using alpha = sqrt(4 ln2) / fwhm.
    Weight = 0.5 * [erf(alpha*(R-Ei)) - erf(alpha*(L-Ei))]
    """
    alpha = math.sqrt(4.0 * math.log(2.0)) / max(1e-12, fwhm)
    # vectorized with python's math.erf via numpy.frompyfunc for stability
    def _erf(v: float) -> float: return math.erf(v)
    uerf = np.frompyfunc(_erf, 1, 1)
    return 0.5 * (uerf(alpha * (R - Ei)).astype(float) - uerf(alpha * (L - Ei)).astype(float))

def assign_states_to_regions(
    df: pd.DataFrame,
    energy_cols: List[str],
    strength_cols: List[str],
    config_col: str,
    regions: List[Region],
    sigma: float = 0.04,
    states_mode: Literal["first", "all"] = "first",
    fractional: bool = False,
    save: bool = True,
    list_excluded: bool = False,
    table_path: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Assign snapshots/states to regions using hard or fractional rules.

    - Hard: a state belongs to the single region whose span [L,R] contains Ei (ties → left).
    - Fractional: a state contributes to all regions with Gaussian overlap weight.

    Outputs
    -------
    - state_assignment.csv
    - region_summary.csv
    """
    cfg = df[config_col].to_numpy(int)
    E = df[energy_cols].to_numpy(float)          # shape (N, S)
    F = df[strength_cols].to_numpy(float)        # shape (N, S)

    if states_mode == "first":
        E = E[:, :1]
        F = F[:, :1]

    N, S = E.shape
    Rn = len(regions)

    # Build weights W of shape (N*S, R)
    Ei = E.reshape(-1)
    fi = F.reshape(-1)

    # build region span arrays
    span_L = np.array([r.left_energy for r in regions], dtype=float)
    span_R = np.array([r.right_energy for r in regions], dtype=float)

    if fractional:
        W = np.zeros((Ei.size, Rn), dtype=float)
        for j in range(Rn):
            wj = _gauss_region_weight(Ei, sigma, span_L[j], span_R[j])
            W[:, j] = np.clip(wj, 0.0, 1.0)
        # normalize per-state so sum_j W_ij ≤ 1 (Gaussian tails outside all regions are discarded)
        sums = W.sum(axis=1, keepdims=True)
        nz = sums[:, 0] > 0
        W[nz, :] = W[nz, :] / sums[nz]
    else:
        # Hard assignment
        centers = 0.5 * (span_L + span_R)
        # choose region whose [L,R] contains Ei, else closest center
        W = np.zeros((Ei.size, Rn), dtype=float)
        for i, e in enumerate(Ei):
            hits = np.where((e >= span_L) & (e <= span_R))[0]
            if hits.size:
                # if multiple, choose the left-most deterministically
                j = int(hits[0])
            else:
                j = int(np.argmin(np.abs(centers - e)))
            W[i, j] = 1.0

    # rows for assignment table
    rows = []
    for idx in range(Ei.size):
        if fi[idx] <= 0:
            continue
        nnz = np.where(W[idx] > 1e-12)[0]
        if nnz.size == 0:
            continue
        n = idx // S
        s = (idx % S) + 1
        for j in nnz:
            w = float(W[idx, j])
            if w <= 0:
                continue
            rows.append({
                "Configuration": int(cfg[n]),
                "state": int(s),
                "region": int(j + 1),
                "Ei": float(Ei[idx]),
                "fi": float(fi[idx]),
                "weight": w,
                "fi_weighted": float(fi[idx] * w),
            })

    assign_df = pd.DataFrame(rows).sort_values(["region", "Configuration", "state"]).reset_index(drop=True)

    # Region summary
    summary_rows = []
    for j, r in enumerate(regions, start=1):
        sub = assign_df[assign_df["region"] == j]
        if fractional:
            eff_states = sub["weight"].sum()
            f_wsum = sub["fi_weighted"].sum()
            summary_rows.append({
                "region": j,
                "left_e": r.left_energy,
                "right_e": r.right_energy,
                "peak_e": r.peak_energy,
                "peak_intensity": r.peak_intensity,
                "effective_states": eff_states,
                "unique_snapshots": sub["Configuration"].nunique(),
                "f_weighted_sum": f_wsum,
            })
        else:
            summary_rows.append({
                "region": j,
                "left_e": r.left_energy,
                "right_e": r.right_energy,
                "peak_e": r.peak_energy,
                "peak_intensity": r.peak_intensity,
                "n_states": len(sub),
                "unique_snapshots": sub["Configuration"].nunique(),
                "f_sum": sub["fi"].sum(),
            })
    summary_df = pd.DataFrame(summary_rows)

    # Save
    if save and table_path:
        base = Path(table_path).with_suffix("")
        assign_csv = f"{base.name}_state_assignment.csv"
        summary_csv = f"{base.name}_region_summary.csv"
        assign_df.to_csv(assign_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Wrote {assign_csv}")
        logger.info(f"Wrote {summary_csv}")

    # Excluded list: leave to extractor (we maintain all qualifying snapshots here)
    if list_excluded and table_path:
        ex_path = f"{Path(table_path).with_suffix('').name}_excluded_configurations.txt"
        Path(ex_path).write_text("")  # placeholder; extraction already lists excluded
        logger.info(f"Wrote excluded configuration IDs → {ex_path}")

    return {"assignments": assign_df, "summary": summary_df}
