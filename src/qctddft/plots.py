from __future__ import annotations
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from .regions import Region

def plot_spectrum(x: np.ndarray, y: np.ndarray, save_path: str | None = None):
    """
    Generates and optionally saves a simple plot of a spectrum.
    """
    plt.figure()
    plt.plot(x, y, lw=1.2)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Intensity (arb. u.)")
    plt.title("Convoluted Spectrum")
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
    