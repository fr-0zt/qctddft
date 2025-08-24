from __future__ import annotations
import re
import os
import glob
from typing import Iterable, List, Tuple, Optional
import pandas as pd

# We don’t import the package logger at import time to keep this file usable standalone.
try:
    from . import logger
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger("qctddft")
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# -----------------------------
# Regex patterns for Q-Chem TDDFT
# -----------------------------
# EFP (polarization-corrected energies)
_EFP_ENERGY_RE = re.compile(
    r"Excitation energy with pol correction \(in eV\)\s*=\s*([-]?\d+\.\d+)"
)
# PCM/VAC (standard TD-DFT excited-state line)
_PCM_ENERGY_RE = re.compile(
    r"Excited state\s+\d+:\s+excitation energy \(eV\)\s*=\s*([-]?\d+\.\d+)"
)
# Common for both
_STRENGTH_RE = re.compile(r"Strength\s*:\s*([0-9]+\.[0-9]+)")

# config id: prefer directory .../so3sq_<n>/..., else any so3sq_<n> in basename
_CFG_FROM_DIR_RE = re.compile(r"(?:^|/)so3sq[_-](\d+)(?:/|$)")
_CFG_FROM_FILE_RE = re.compile(r"so3sq[_-](\d+)")


def _extract_config_id(path: str) -> Optional[int]:
    """
    Resolve configuration number from a Q-Chem TDDFT output path.
    Works for both:
      - TDDFT/TDDFT_EFP/so3sq_99/so3sq-water_..._99_tddft.out
      - TDDFT/TDDFT_PCM/so3sq_1/so3sq_1_tddft.out
    """
    m = _CFG_FROM_DIR_RE.search(path)
    if m:
        return int(m.group(1))
    m = _CFG_FROM_FILE_RE.search(os.path.basename(path))
    if m:
        return int(m.group(1))
    return None


def _parse_qchem_tddft_block(
    lines: Iterable[str],
    env: str,
    max_states: int,
) -> Tuple[List[float], List[float]]:
    """
    Parse the TDDFT section for either EFP or PCM/VAC.
    Returns (energies, strengths) lists (possibly of different length; we align later).
    """
    energies: List[float] = []
    strengths: List[float] = []

    # Start capture after seeing the TDDFT header
    capture = False
    for ln in lines:
        if "TDDFT Excitation Energies" in ln:
            capture = True
            continue
        if not capture:
            continue

        if env == "efp":
            em = _EFP_ENERGY_RE.search(ln)
        else:  # pcm or vac
            em = _PCM_ENERGY_RE.search(ln)

        sm = _STRENGTH_RE.search(ln)

        if em:
            energies.append(float(em.group(1)))
        if sm:
            strengths.append(float(sm.group(1)))

        # quick exit if both hit max
        if len(energies) >= max_states and len(strengths) >= max_states:
            break

    return energies, strengths


def _pair_and_filter_first_state(
    energies: List[float],
    strengths: List[float],
    max_states: int,
    f1_min: float,
) -> List[Tuple[float, float]]:
    """
    Zip energies & strengths (truncate to min length), apply first-state inclusion filter,
    then return up to max_states pairs.
    """
    n = min(len(energies), len(strengths))
    if n == 0:
        return []

    pairs = list(zip(energies[:n], strengths[:n]))
    # first-state inclusion rule
    e1, f1 = pairs[0]
    if not (e1 > 0.0 and f1 >= f1_min):
        return []
    return pairs[:max_states]


def extract_directory(
    input_glob: str,
    *,
    max_states: int = 5,
    f1_min: float = 0.10,
    env: str = "auto",         # 'auto' | 'efp' | 'pcm' | 'vac'
    strict_env: bool = False,  # if True, error when 'auto' can’t decide (we default pcm/vac)
    config_min: Optional[int] = None,
    config_max: Optional[int] = None,
) -> pd.DataFrame:
    """
    Scan a glob of Q-Chem TDDFT output files and produce a tidy table:
    columns: Configuration, Energy 1..N, Strength 1..N

    Parameters
    ----------
    input_glob : glob pattern of .out files
    max_states : maximum states to collect
    f1_min : first-state oscillator strength threshold (E1>0 & f1>=f1_min)
    env : 'efp' uses polarization-corrected energy line; 'pcm' and 'vac' use standard line.
          If 'auto', pick by sniffing the presence of the EFP line.
    strict_env : if True and auto detection fails, raise; else fall back to 'pcm'.
    config_min/config_max : optional inclusive range of configuration IDs to keep
    """
    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No files match: {input_glob}")

    rows: List[dict] = []
    dropped_missing_cfg = 0
    dropped_by_first_state = 0

    for p in paths:
        cfg = _extract_config_id(p)
        if cfg is None:
            logger.warning("Could not identify configuration number from %s; skipping.", p)
            dropped_missing_cfg += 1
            continue

        if config_min is not None and cfg < config_min:
            continue
        if config_max is not None and cfg > config_max:
            continue

        with open(p, "r", errors="replace") as fh:
            lines = fh.readlines()

        # Decide env if auto
        env_here = env
        if env_here == "auto":
            # If we ever see the EFP energy-with-pol-correction line in file, treat as EFP
            if any(_EFP_ENERGY_RE.search(ln) for ln in lines):
                env_here = "efp"
            else:
                env_here = "pcm"  # default fallback for non-EFP outputs
                if strict_env:
                    raise RuntimeError(f"Could not confirm 'efp' in {p}; strict_env=True and auto failed.")

        energies, strengths = _parse_qchem_tddft_block(lines, env_here, max_states)
        pairs = _pair_and_filter_first_state(energies, strengths, max_states, f1_min=f1_min)
        if not pairs:
            dropped_by_first_state += 1
            continue

        # pad to max_states
        while len(pairs) < max_states:
            pairs.append((0.0, 0.0))

        row = {"Configuration": int(cfg)}
        for i, (e, f) in enumerate(pairs, start=1):
            row[f"Energy {i}"] = float(e)
            row[f"Strength {i}"] = float(f)
        rows.append(row)

    if not rows:
        raise RuntimeError("No snapshots passed the first-state inclusion rule.")

    if dropped_missing_cfg:
        logger.warning("Skipped %d files due to missing config id.", dropped_missing_cfg)
    if dropped_by_first_state:
        logger.info("Excluded %d snapshots by first-state rule (E1>0 & f1>=%.3g).",
                    dropped_by_first_state, f1_min)

    df = pd.DataFrame(rows).sort_values("Configuration").reset_index(drop=True)
    return df
