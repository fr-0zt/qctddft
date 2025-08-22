# src/qctddft/regions.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np

__all__ = [
    "Region",
    "identify_regions",
    "plot_regions",
]

Selection = Literal["top", "centered"]  # how to keep N regions

@dataclass(frozen=True)
class Region:
    left_idx: int
    right_idx: int
    left_energy: float
    right_energy: float
    peak_idx: int
    peak_energy: float
    peak_intensity: float

# ---------- calculus ----------
def _gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.gradient(y, x)

def _second_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.gradient(_gradient(x, y), x)

def _local_maxima(arr: np.ndarray) -> np.ndarray:
    if arr.size < 3:
        return np.empty(0, dtype=int)
    mid = (arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:])
    return np.where(mid)[0] + 1

# ---------- boundaries ----------
def _pick_boundary_indices(
    d2y: np.ndarray,
    k_min: int = 4,
    prominence_quantile: float = 0.60,
) -> List[int]:
    """
    Choose boundary candidates at local maxima of d²I/dE².
    Keep those above a quantile threshold; ensure at least k_min by falling
    back to the strongest maxima.
    """
    cand = _local_maxima(d2y)
    if cand.size == 0:
        return []
    thr = float(np.quantile(d2y[cand], np.clip(prominence_quantile, 0.05, 0.95)))
    picks = cand[d2y[cand] >= thr]
    if picks.size < k_min:
        order = cand[np.argsort(d2y[cand])[::-1]]
        picks = order[:k_min]
    return sorted(set(picks.tolist()))

def _regions_from_boundaries_raw(
    x: np.ndarray,
    y: np.ndarray,
    boundary_idx: List[int],
    min_width_pts: int = 3,
) -> List[Region]:
    if len(boundary_idx) < 2:
        l, r = 0, len(x) - 1
        pk = l + int(np.argmax(y[l:r+1]))
        return [Region(l, r, x[l], x[r], pk, x[pk], float(y[pk]))]

    b = sorted(boundary_idx)
    if b[0] != 0:
        b.insert(0, 0)
    nlast = len(x) - 1
    if b[-1] != nlast:
        b.append(nlast)

    regs: List[Region] = []
    for i in range(len(b) - 1):
        l, r = b[i], b[i + 1]
        if r - l + 1 < min_width_pts:
            continue
        pk = l + int(np.argmax(y[l:r+1]))
        regs.append(
            Region(
                left_idx=l,
                right_idx=r,
                left_energy=float(x[l]),
                right_energy=float(x[r]),
                peak_idx=pk,
                peak_energy=float(x[pk]),
                peak_intensity=float(y[pk]),
            )
        )
    regs.sort(key=lambda R: R.peak_energy)
    return regs

# ---------- merging slivers (energy space) ----------
def _merge_edge_slivers(regs: List[Region], min_width_e: float) -> List[Region]:
    if not regs or min_width_e <= 0:
        return regs[:]
    out = regs[:]
    changed = True
    while changed and len(out) >= 2:
        changed = False
        # left edge
        r0 = out[0]
        if (r0.right_energy - r0.left_energy) < min_width_e:
            r1 = out[1]
            pk = r0 if r0.peak_intensity >= r1.peak_intensity else r1
            out = [
                Region(
                    left_idx=r0.left_idx,
                    right_idx=r1.right_idx,
                    left_energy=r0.left_energy,
                    right_energy=r1.right_energy,
                    peak_idx=pk.peak_idx,
                    peak_energy=pk.peak_energy,
                    peak_intensity=pk.peak_intensity,
                )
            ] + out[2:]
            changed = True
            continue
        # right edge
        rn = out[-1]
        if (rn.right_energy - rn.left_energy) < min_width_e:
            rm = out[-2]
            pk = rn if rn.peak_intensity >= rm.peak_intensity else rm
            out = out[:-2] + [
                Region(
                    left_idx=rm.left_idx,
                    right_idx=rn.right_idx,
                    left_energy=rm.left_energy,
                    right_energy=rn.right_energy,
                    peak_idx=pk.peak_idx,
                    peak_energy=pk.peak_energy,
                    peak_intensity=pk.peak_intensity,
                )
            ]
            changed = True
            continue
    return out

def _merge_internal_slivers_if_weak(
    regs: List[Region],
    min_width_e: float,
    weak_ratio: float = 0.65,
) -> List[Region]:
    if min_width_e <= 0 or not regs:
        return regs[:]
    out: List[Region] = []
    i = 0
    while i < len(regs):
        r = regs[i]
        width = r.right_energy - r.left_energy
        if 0 < i < len(regs) - 1 and width < min_width_e:
            left_pk = regs[i - 1].peak_intensity
            right_pk = regs[i + 1].peak_intensity
            # merge into stronger neighbor (prefer right on ties)
            if r.peak_intensity < weak_ratio * max(left_pk, right_pk):
                if right_pk >= left_pk:
                    n = regs[i + 1]
                    pk = r if r.peak_intensity >= n.peak_intensity else n
                    out.append(
                        Region(
                            left_idx=r.left_idx,
                            right_idx=n.right_idx,
                            left_energy=r.left_energy,
                            right_energy=n.right_energy,
                            peak_idx=pk.peak_idx,
                            peak_energy=pk.peak_energy,
                            peak_intensity=pk.peak_intensity,
                        )
                    )
                    i += 2
                    continue
                else:
                    p = out.pop() if out else regs[i - 1]
                    pk = r if r.peak_intensity >= p.peak_intensity else p
                    out.append(
                        Region(
                            left_idx=p.left_idx,
                            right_idx=r.right_idx,
                            left_energy=p.left_energy,
                            right_energy=r.right_energy,
                            peak_idx=pk.peak_idx,
                            peak_energy=pk.peak_energy,
                            peak_intensity=pk.peak_intensity,
                        )
                    )
                    i += 1
                    continue
        out.append(r)
        i += 1
    out.sort(key=lambda R: R.peak_energy)
    return out

# ---------- selection of N ----------
def _select_top_n(regs: List[Region], n: int) -> List[Region]:
    regs = sorted(regs, key=lambda r: r.peak_intensity, reverse=True)[:n]
    return sorted(regs, key=lambda r: r.peak_energy)

def _select_centered_n(regs: List[Region], n: int) -> List[Region]:
    if len(regs) <= n:
        return regs[:]
    regs_sorted = sorted(regs, key=lambda r: r.peak_energy)
    kmax = int(np.argmax([r.peak_intensity for r in regs_sorted]))
    picks = {kmax}
    L, R = kmax - 1, kmax + 1
    while len(picks) < n and (L >= 0 or R < len(regs_sorted)):
        if L >= 0:
            picks.add(L)
            L -= 1
            if len(picks) == n:
                break
        if R < len(regs_sorted):
            picks.add(R)
            R += 1
    return [regs_sorted[i] for i in sorted(picks)]

# ---------- public API ----------
def identify_regions(
    energies: np.ndarray,
    intensities: np.ndarray,
    n_regions: int = 3,
    prominence_quantile: float = 0.60,
    min_width_e: float = 0.035,
    selection: Selection = "top",
    y_floor_frac: float = 0.0,
) -> Tuple[List[Region], Dict[str, np.ndarray]]:
    """
    Identify peak/shoulder regions using 2nd-derivative maxima as boundaries (no smoothing).

    Parameters
    ----------
    energies : 1D array (strictly increasing)
    intensities : 1D array
    n_regions : number of regions to retain
    prominence_quantile : quantile threshold applied to local maxima of d²I/dE²
    min_width_e : minimal allowed region width (eV) — slivers are merged
    selection : 'top' (by intensity) or 'centered' (around the strongest peak)
    y_floor_frac : optional baseline removal; subtract y_floor_frac * max(intensity)

    Returns
    -------
    regions : list[Region]
    arrays  : dict with "dy", "d2y", "boundary_idx"
    """
    x = np.asarray(energies, float)
    y = np.asarray(intensities, float)
    assert x.ndim == 1 and y.ndim == 1 and x.size == y.size and x.size > 5
    assert np.all(np.diff(x) > 0), "energies must be strictly increasing"

    if y_floor_frac > 0:
        y = y - float(y_floor_frac) * float(np.max(y))
        y = np.clip(y, 0.0, None)

    dy = _gradient(x, y)
    d2y = _second_derivative(x, y)
    bidx = _pick_boundary_indices(d2y, k_min=4, prominence_quantile=prominence_quantile)

    regs = _regions_from_boundaries_raw(x, y, bidx, min_width_pts=3)
    if not regs:
        return [], {"dy": dy, "d2y": d2y, "boundary_idx": np.array(bidx, int)}

    # merge slivers (edge first, then internal if weak)
    regs = _merge_edge_slivers(regs, min_width_e)
    regs = _merge_internal_slivers_if_weak(regs, min_width_e, weak_ratio=0.65)

    # keep N
    if len(regs) > n_regions:
        regs = _select_top_n(regs, n_regions) if selection == "top" else _select_centered_n(regs, n_regions)

    regs.sort(key=lambda r: r.peak_energy)
    return regs, {"dy": dy, "d2y": d2y, "boundary_idx": np.array(bidx, int)}

# ---------- plotting ----------
def plot_regions(
    energies: np.ndarray,
    intensities: np.ndarray,
    regions: List[Region],
    arrays: Dict[str, np.ndarray],
    show_d2: bool = True,
    show_only_used_boundaries: bool = True,
    title: str = "Convoluted spectrum with regions (2nd-derivative boundaries)",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting; install qctddft[plot]") from exc

    x = np.asarray(energies, float)
    y = np.asarray(intensities, float)
    d2y = arrays.get("d2y")
    all_bidx = np.asarray(arrays.get("boundary_idx", []), int)

    # draw only region edges (avoid faint extra lines)
    if show_only_used_boundaries and regions:
        used = [regions[0].left_idx] + [r.left_idx for r in regions[1:]] + [regions[-1].right_idx]
        bidx = np.array(sorted(set(used)), int)
    else:
        bidx = all_bidx

    # Figure 1: spectrum
    plt.figure()
    plt.plot(x, y, lw=1.2)
    for r in regions:
        plt.axvspan(r.left_energy, r.right_energy, alpha=0.15)
        plt.axvline(x[r.peak_idx], linestyle="--", alpha=0.7)
    for i in bidx:
        plt.axvline(x[i], alpha=0.3)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Intensity (arb. u.)")
    plt.title(title)
    plt.show()

    # Figure 2: d²I/dE²
    if show_d2 and d2y is not None:
        plt.figure()
        plt.plot(x, d2y, lw=1.0)
        for i in bidx:
            plt.axvline(x[i], alpha=0.3)
        plt.xlabel("Energy (eV)")
        plt.ylabel("d²I/dE² (arb. u.)")
        plt.title("Second derivative with boundary maxima")
        plt.show()
