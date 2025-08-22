from __future__ import annotations
import re, os, glob
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal
import pandas as pd

Env = Literal["efp", "pcm", "vacuum", "auto"]

# ---------------- Patterns ----------------
HDR_TDDFT = re.compile(r"TDDFT\s+Excitation\s+Energies", re.IGNORECASE)

# PCM/Vacuum canonical energy + inline f
PCM_ENERGY = re.compile(
    r"Excited\s+state\s+(\d+)\s*:\s*excitation\s+energy\s*\(eV\)\s*=\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
INLINE_F = re.compile(r"\bf\s*=\s*([-+]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\b")

# EFP polarization-corrected energy lines
EFP_ENERGY = re.compile(
    r"Excitation\s+energy\s+with\s+pol\s+correction\s*\(in\s*eV\)\s*=\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Generic strength lines (often appear on the next line in some formats)
LINE_STRENGTH = re.compile(
    r"Strength\s*:\s*([-+]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)",
    re.IGNORECASE,
)

# EFP job markers to help auto-detect
EFP_MARKERS = [
    re.compile(r"\bEFP\b", re.IGNORECASE),
    re.compile(r"Effective\s+Fragment\s+Potential", re.IGNORECASE),
    re.compile(r"Polarization\s+correction", re.IGNORECASE),
]

# ---------------- Data structures ----------------
@dataclass
class TDDFTRecord:
    config: int
    energies: List[float]
    strengths: List[float]

# ---------------- Utilities ----------------
def infer_config_id(path: str) -> int:
    """Infer configuration ID robustly from file/dir names."""
    # Common: .../so3sq_123/...
    m = re.findall(r"so3sq[_\-]?(\d+)", path, flags=re.IGNORECASE)
    if m: return int(m[-1])
    # Fallback: last integer in the path
    m2 = re.findall(r"(\d+)", path)
    return int(m2[-1]) if m2 else -1

def first_state_passes(e: List[float], f: List[float], f1_min: float) -> bool:
    return bool(e) and bool(f) and (e[0] > 0.0) and (f[0] >= f1_min)

def _looks_like_efp(text: str) -> bool:
    if EFP_ENERGY.search(text):  # strongest signal
        return True
    return any(p.search(text) for p in EFP_MARKERS)

# ---------------- Core parsing ----------------
def parse_qchem_tddft(path: str, max_states: int, env: Env) -> Optional[Tuple[List[float], List[float]]]:
    """
    Parse a single Q-Chem TDDFT output for up to `max_states` states.

    Modes:
      - 'efp'   : read ONLY polarization-corrected energies (EFP)
      - 'pcm'   : read ONLY canonical 'Excited state ... excitation energy (eV) = ...'
      - 'vacuum': same as 'pcm'
      - 'auto'  : detect from file text; prefer EFP if pol-corrected energies are present

    Returns (energies, strengths) or None if nothing found.
    """
    try:
        with open(path, "r", errors="ignore") as fh:
            text = fh.read()
    except OSError:
        return None

    effective_env: Env = env
    if env == "auto":
        effective_env = "efp" if _looks_like_efp(text) else "pcm"

    # Work only within the TDDFT section to avoid matching spurious lines elsewhere.
    sect = HDR_TDDFT.split(text, maxsplit=1)
    if len(sect) < 2:
        return None
    body = sect[1]

    energies_map: dict[int, float] = {}
    strengths_map: dict[int, float] = {}

    if effective_env == "efp":
        # EFP: states are implied by order of pol-corrected lines (no explicit indices)
        e_lines = EFP_ENERGY.findall(body)
        # Some outputs print strengths near-by; use a one-pass scan to align strengths to energies.
        # Strategy: collect energies in order; then walk line-by-line and assign strengths to the first state without f.
        if not e_lines:
            return None
        e_vals: List[float] = [float(v) for v in e_lines[:max_states]]
        # Now assign strengths
        strengths: List[float] = [0.0] * len(e_vals)
        # line-by-line pass to fill f in order encountered
        pending = 0
        for line in body.splitlines():
            if EFP_ENERGY.search(line):
                # every time we hit an energy line, we allow the next Strength line to attach
                continue
            ms = LINE_STRENGTH.search(line)
            if ms and pending < len(strengths):
                try:
                    strengths[pending] = float(ms.group(1))
                except ValueError:
                    pass
                pending += 1
                if pending >= len(strengths):
                    break
        energies = e_vals + [0.0] * max(0, max_states - len(e_vals))
        strengths = strengths + [0.0] * max(0, max_states - len(strengths))
        return energies[:max_states], strengths[:max_states]

    else:
        # PCM/Vacuum canonical lines with explicit state index
        # We also capture inline f on the same line, then fill any missing f from 'Strength:' lines that follow.
        energies_map.clear(); strengths_map.clear()
        for line in body.splitlines():
            mE = PCM_ENERGY.search(line)
            if mE:
                st = int(mE.group(1))
                if st <= max_states:
                    energies_map[st] = float(mE.group(2))
                    # inline f?
                    mf = INLINE_F.search(line)
                    if mf:
                        strengths_map[st] = float(mf.group(1))
                continue
            ms = LINE_STRENGTH.search(line)
            if ms:
                # attach to lowest state that has energy but no f yet
                open_states = sorted(k for k in energies_map if k not in strengths_map)
                if open_states:
                    strengths_map[open_states[0]] = float(ms.group(1))

        if not energies_map:
            return None
        energies = [float(energies_map.get(i, 0.0)) for i in range(1, max_states+1)]
        strengths = [float(strengths_map.get(i, 0.0)) for i in range(1, max_states+1)]
        return energies, strengths

# ---------------- Public API ----------------
def extract_directory(
    input_glob: str,
    max_states: int = 5,
    f1_min: float = 0.1,
    env: Env = "auto",
    strict_env: bool = False,
) -> pd.DataFrame:
    """
    Scan TDDFT outputs and produce a wide table:
      Configuration, Energy1..N, Strength1..N

    - env:
        'efp'    → only pol-corrected energies
        'pcm'    → only canonical TDDFT energies
        'vacuum' → same as 'pcm'
        'auto'   → inspect text; choose efp if pol-corrected lines present, else pcm
    - strict_env:
        If True, files that do not match the required pattern for the chosen env are skipped.
        If False, they will just parse whatever is present (useful when mixed).
    """
    files = sorted(glob.glob(input_glob, recursive=True))
    if not files:
        raise FileNotFoundError(f"No files matched: {input_glob}")

    rows: List[dict] = []
    for fp in files:
        parsed = parse_qchem_tddft(fp, max_states=max_states, env=env)
        if not parsed:
            if strict_env:
                continue
            # try the other mode as a fallback when in 'auto' or non-strict
            if env == "efp":
                parsed = parse_qchem_tddft(fp, max_states=max_states, env="pcm")
            elif env in ("pcm", "vacuum"):
                parsed = parse_qchem_tddft(fp, max_states=max_states, env="efp")
            elif env == "auto":
                parsed = parse_qchem_tddft(fp, max_states=max_states, env="pcm")
            if not parsed:
                continue

        e, f = parsed
        if not first_state_passes(e, f, f1_min):
            continue

        cfg = infer_config_id(fp)
        row = {"Configuration": cfg}
        for i in range(1, max_states + 1):
            row[f"Energy{i}"] = float(e[i-1]) if i-1 < len(e) else 0.0
            row[f"Strength{i}"] = float(f[i-1]) if i-1 < len(f) else 0.0
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["Configuration"] +
                            [f"Energy{i}" for i in range(1, max_states+1)] +
                            [f"Strength{i}" for i in range(1, max_states+1)])

    df = pd.DataFrame(rows).sort_values("Configuration").reset_index(drop=True)
    return df

def to_wide_csv(df: pd.DataFrame, out_csv: str, max_states: int) -> None:
    cols = ["Configuration"] + [f"Energy{i}" for i in range(1, max_states+1)] + [f"Strength{i}" for i in range(1, max_states+1)]
    df = df.reindex(columns=cols, fill_value=0.0)
    df.to_csv(out_csv, index=False)
