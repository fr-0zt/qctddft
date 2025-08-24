from __future__ import annotations
import numpy as np
from typing import List, Dict
from .regions import Region, plot_regions as _plot_regions

def plot_regions(
    x: np.ndarray,
    y: np.ndarray,
    regions: List[Region],
    arrays: Dict[str, np.ndarray],
    **kwargs,
) -> None:
    _plot_regions(x, y, regions, arrays, **kwargs)
