from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass(frozen=True)
class Region:
    left_idx: int
    right_idx: int
    left_energy: float
    right_energy: float
    peak_idx: int
    peak_energy: float
    peak_intensity: float

def _gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.gradient(y, x)

def _second_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.gradient(_gradient(x, y), x)

def _local_maxima(arr: np.ndarray) -> np.ndarray:
    if arr is None or len(arr) < 3: return np.array([], dtype=int)
    return np.where((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]))[0] + 1

def _pick_boundary_indices(d2y: np.ndarray, k_min: int = 2, q: float = 0.60) -> List[int]:
    cand = _local_maxima(d2y)
    if cand.size == 0: return []
    thr = float(np.quantile(d2y[cand], q))
    picks = cand[d2y[cand] >= thr].tolist()
    if len(picks) < k_min:
        order = cand[np.argsort(d2y[cand])[::-1]]
        picks = order[:k_min].tolist()
    return sorted(set(picks))

def _regions_from_boundaries(x: np.ndarray, y: np.ndarray, bidx: List[int], min_pts: int = 3) -> List[Region]:
    if len(bidx) < 2:
        l, r = 0, len(x)-1
        pk = int(np.argmax(y[l:r+1])) + l
        return [Region(l, r, x[l], x[r], pk, x[pk], y[pk])]
    b = sorted(bidx)
    if b[0] != 0: b = [0] + b
    if b[-1] != len(x)-1: b = b + [len(x)-1]
    out: List[Region] = []
    for i in range(len(b)-1):
        l, r = b[i], b[i+1]
        if r - l + 1 < min_pts: continue
        pk = l + int(np.argmax(y[l:r+1]))
        out.append(Region(l, r, x[l], x[r], pk, x[pk], y[pk]))
    return out

def _merge_narrow_regions(regs: List[Region], min_w_e: float) -> List[Region]:
    if not regs or min_w_e <= 0: return regs
    regs = sorted(regs, key=lambda r: r.peak_energy)
    def width(r: Region) -> float: return r.right_energy - r.left_energy
    changed = True
    while changed and len(regs) > 1:
        changed = False
        for i, r in enumerate(regs):
            if width(r) < min_w_e:
                if i == 0:
                    a, b = regs[0], regs[1]
                    pk = a if a.peak_intensity >= b.peak_intensity else b
                    regs = [Region(a.left_idx, b.right_idx, a.left_energy, b.right_energy,
                                   pk.peak_idx, pk.peak_energy, pk.peak_intensity)] + regs[2:]
                elif i == len(regs)-1:
                    a, b = regs[-2], regs[-1]
                    pk = a if a.peak_intensity >= b.peak_intensity else b
                    regs = regs[:-2] + [Region(a.left_idx, b.right_idx, a.left_energy, b.right_energy,
                                               pk.peak_idx, pk.peak_energy, pk.peak_intensity)]
                else:
                    L, C, R = regs[i-1], regs[i], regs[i+1]
                    left_m  = Region(L.left_idx, C.right_idx, L.left_energy, C.right_energy,
                                     (L if L.peak_intensity >= C.peak_intensity else C).peak_idx,
                                     (L if L.peak_intensity >= C.peak_intensity else C).peak_energy,
                                     max(L.peak_intensity, C.peak_intensity))
                    right_m = Region(C.left_idx, R.right_idx, C.left_energy, R.right_energy,
                                     (C if C.peak_intensity >= R.peak_intensity else R).peak_idx,
                                     (C if C.peak_intensity >= R.peak_intensity else R).peak_energy,
                                     max(C.peak_intensity, R.peak_intensity))
                    # prefer wider merge
                    if (right_m.right_energy - right_m.left_energy) >= (left_m.right_energy - left_m.left_energy):
                        regs = regs[:i] + [right_m] + regs[i+2:]
                    else:
                        regs = regs[:i-1] + [left_m] + regs[i+1:]
                regs = sorted(regs, key=lambda rr: rr.peak_energy)
                changed = True
                break
    return regs

def _select_centered_n(regs: List[Region], n: int) -> List[Region]:
    regs = sorted(regs, key=lambda r: r.peak_energy)
    if len(regs) <= n: return regs
    k = int(np.argmax([r.peak_intensity for r in regs]))
    picks, L, R = {k}, k-1, k+1
    while len(picks) < n and (L >= 0 or R < len(regs)):
        if L >= 0: picks.add(L); L -= 1
        if len(picks) == n: break
        if R < len(regs): picks.add(R); R += 1
    return [regs[i] for i in sorted(picks)]

def _select_top_intensity(regs: List[Region], n: int) -> List[Region]:
    regs_sorted = sorted(regs, key=lambda r: (r.peak_intensity, r.right_energy - r.left_energy, -r.peak_energy), reverse=True)
    picked = regs_sorted[:n]
    return sorted(picked, key=lambda r: r.peak_energy)

def identify_regions(
    energies: np.ndarray,
    intensities: np.ndarray,
    n_regions: int = 3,
    prom_q: float = 0.60,       # alias for prominence_quantile
    k_min: int = 4,
    min_width_e: float = 0.02,
    min_width_pts: int = 3,
    selection: str = "top",     # "top" or "centered"
    y_floor_frac: float = 0.00,
    band_area_q: float = 0.00,
    **kwargs,
) -> Tuple[List[Region], Dict[str, np.ndarray]]:
    prominence_quantile = float(kwargs.get("prominence_quantile", prom_q))

    x = np.asarray(energies, float)
    y = np.asarray(intensities, float)
    assert x.ndim == y.ndim == 1 and x.size == y.size and x.size > 5
    assert np.all(np.diff(x) > 0), "energies must be strictly increasing"

    # Optional trimming/tail handling
    if y_floor_frac > 0.0:
        ymax = float(np.max(y))
        thr = ymax * y_floor_frac
        mask = y >= thr
        if np.any(mask):
            k0 = int(np.argmax(y))
            L = k0
            while L > 0 and mask[L-1]: L -= 1
            R = k0
            while R < len(y)-1 and mask[R+1]: R += 1
            x, y = x[L:R+1], y[L:R+1]

    if band_area_q > 0.0:
        y_pos = np.clip(y, 0.0, None)
        area = float(np.trapezoid(y_pos, x))
        if area > 0:
            c = np.cumsum((y_pos[1:]+y_pos[:-1]) * 0.5 * np.diff(x))
            c = np.concatenate([[0.0], c]) / area
            lo = float(np.clip(band_area_q, 0.0, 0.45)); hi = 1.0 - lo
            i_lo = int(np.searchsorted(c, lo))
            i_hi = int(np.searchsorted(c, hi))
            i_lo = max(0, min(i_lo, len(x)-2))
            i_hi = max(i_lo+1, min(i_hi, len(x)-1))
            x, y = x[i_lo:i_hi+1], y[i_lo:i_hi+1]

    # curvature + boundaries
    d2y = _second_derivative(x, y)
    bidx = _pick_boundary_indices(d2y, k_min=max(2, int(k_min)), q=prominence_quantile)

    regs = _regions_from_boundaries(x, y, bidx, min_pts=min_width_pts)
    regs = _merge_narrow_regions(regs, min_width_e)

    if len(regs) > n_regions:
        regs = _select_top_intensity(regs, n_regions) if selection == "top" else _select_centered_n(regs)
    regs = sorted(regs, key=lambda r: r.peak_energy)

    return regs, {"x": x, "y": y, "d2y": d2y, "boundary_idx": np.array(bidx, int)}
"""
def plot_regions(
    energies: np.ndarray,
    intensities: np.ndarray,
    regions: List[Region],
    arrays: Dict[str, np.ndarray],
    show_d2: bool = True,
    save_path_spectrum: Optional[str] = None,
    save_path_d2: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    x = np.asarray(arrays.get("x", energies), float)
    y = np.asarray(arrays.get("y", intensities), float)
    d2y = arrays.get("d2y")

    if regions:
        edges = [regions[0].left_energy] + [r.left_energy for r in regions[1:]] + [regions[-1].right_energy]
    else:
        edges = []

    plt.figure()
    plt.plot(x, y, lw=1.2)
    for r in regions:
        plt.axvspan(r.left_energy, r.right_energy, alpha=0.15)
        plt.axvline(r.peak_energy, linestyle="--", alpha=0.8)
    for xv in edges:
        plt.axvline(xv, alpha=0.35)
    plt.xlabel("Energy (eV)"); plt.ylabel("Intensity (arb. u.)")
    plt.title("Convoluted spectrum with regions")
    if save_path_spectrum: plt.savefig(save_path_spectrum, dpi=200, bbox_inches="tight")
    plt.show()

    if show_d2 and d2y is not None:
        plt.figure()
        plt.plot(x, d2y, lw=1.0)
        for xv in edges:
            plt.axvline(xv, alpha=0.35)
        plt.xlabel("Energy (eV)"); plt.ylabel("d²I/dE² (arb. u.)")
        plt.title("Second derivative with region edges")
        if save_path_d2: plt.savefig(save_path_d2, dpi=200, bbox_inches="tight")
        plt.show()
"""