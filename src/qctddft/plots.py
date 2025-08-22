from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from .regions import Region

def plot_spectrum(x: np.ndarray, y: np.ndarray, label: str = None, save: str | None = None, show: bool = True):
    plt.figure(figsize=(7,4.5))
    plt.plot(x, y, label=(label or "spectrum"))
    plt.xlabel("Energy (eV)"); plt.ylabel("Intensity (arb. u.)"); 
    if label: plt.legend()
    plt.tight_layout()
    if save: plt.savefig(save, dpi=200, bbox_inches="tight")
    if show: plt.show()

def plot_regions(x, y, regions: List[Region], arrays: Dict[str, np.ndarray], show_d2=True, only_used=True, save=None, save_d2=None):
    d2y = arrays.get("d2y"); bidx_all = arrays.get("bidx", np.array([], int))
    if only_used and regions:
        used = {regions[0].left_idx, regions[-1].right_idx}
        for r in regions[1:]:
            used.add(r.left_idx)
        bidx = np.array(sorted(used), int)
    else:
        bidx = bidx_all

    plt.figure(figsize=(7,4.5))
    plt.plot(x, y, lw=1.2)
    for r in regions:
        plt.axvspan(r.left_energy, r.right_energy, alpha=0.15)
        plt.axvline(x[r.peak_idx], linestyle="--", alpha=0.7)
    for i in bidx:
        plt.axvline(x[i], alpha=0.35)
    plt.xlabel("Energy (eV)"); plt.ylabel("Intensity (arb. u.)")
    plt.title("Convoluted spectrum with regions")
    if save: plt.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()

    if show_d2 and d2y is not None:
        plt.figure(figsize=(7,4.5))
        plt.plot(x, d2y, lw=1.0)
        for i in bidx:
            plt.axvline(x[i], alpha=0.35)
        plt.xlabel("Energy (eV)"); plt.ylabel("d²I/dE² (arb. u.)")
        plt.title("Second derivative with boundary maxima")
        if save_d2: plt.savefig(save_d2, dpi=200, bbox_inches="tight")
        plt.show()

