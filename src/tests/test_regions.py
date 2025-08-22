from __future__ import annotations
import numpy as np
from qctddft.regions import identify_regions

def test_three_gaussians():
    x = np.linspace(1.7, 2.4, 2000)
    def g(mu, amp, w):
        return amp*np.exp(-0.5*((x-mu)/w)**2)
    y = g(1.90, 1.0, 0.01) + g(2.02, 2.0, 0.015) + g(2.10, 1.2, 0.012)
    regs, _ = identify_regions(x, y, n_regions=3, prom_q=0.6, k_min=4, min_width_e=0.01)
    assert len(regs) == 3
    peaks = [r.peak_energy for r in regs]
    assert min(abs(p-1.90) for p in peaks) < 0.01
    assert min(abs(p-2.02) for p in peaks) < 0.01
    assert min(abs(p-2.10) for p in peaks) < 0.01

