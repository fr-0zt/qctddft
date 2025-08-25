from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from .regions import Region

def plot_spectrum(x: np.ndarray, y: np.ndarray, save_path: str | None = None):
    """
    Generates and optionally saves a simple plot of a spectrum.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, lw=1.5)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Intensity (arb. u.)")
    plt.title("Convoluted Spectrum")
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()

def plot_regions(
    energies: np.ndarray,
    intensities: np.ndarray,
    regions: List[Region],
    arrays: Dict[str, np.ndarray],
    show_d2: bool = True,
    save_path_spectrum: Optional[str] = None,
    save_path_d2: Optional[str] = None,
):
    """
    Plots the spectrum with highlighted regions and optionally its second derivative.
    """
    # This is the indented block that was missing
    import matplotlib.pyplot as plt
    x = np.asarray(arrays.get("x", energies), float)
    y = np.asarray(arrays.get("y", intensities), float)
    d2y = arrays.get("d2y")

    if regions:
        edges = [regions[0].left_energy] + [r.right_energy for r in regions]
    else:
        edges = []

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, lw=1.5, label="Spectrum")
    for i, r in enumerate(regions, 1):
        plt.axvspan(r.left_energy, r.right_energy, alpha=0.2, label=f"Region {i}")
        plt.axvline(r.peak_energy, color='red', linestyle="--", alpha=0.7)

    plt.xlabel("Energy (eV)")
    plt.ylabel("Intensity (arb. u.)")
    plt.title("Convoluted Spectrum with Identified Regions")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_path_spectrum:
        plt.savefig(save_path_spectrum, dpi=200, bbox_inches="tight")
    plt.show()

    if show_d2 and d2y is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(x, d2y, lw=1.0)
        for r in regions:
             plt.axvline(r.left_energy, color='gray', linestyle='--', alpha=0.5)
             plt.axvline(r.right_energy, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel("Energy (eV)")
        plt.ylabel("d²I/dE² (arb. u.)")
        plt.title("Second Derivative with Region Edges")
        plt.grid(True, linestyle='--', alpha=0.6)
        if save_path_d2:
            plt.savefig(save_path_d2, dpi=200, bbox_inches="tight")
        plt.show()   