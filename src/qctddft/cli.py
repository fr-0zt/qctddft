# src/qctddft/cli.py
from __future__ import annotations
from typing import Optional
import typer
from rich import print as rprint
from rich.table import Table
import pandas as pd

from .extract import extract_directory, to_wide_csv
from .regions import identify_regions, plot_regions

try:
    from . import __version__ as _VERSION
except Exception:
    _VERSION = "0.1.0"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="qctddft: utilities for extracting & organizing Q-Chem TDDFT data.",
)

@app.callback()
def _main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=lambda v: _show_version(v), is_eager=True,
        help="Show package version and exit."
    )
):  # pragma: no cover
    return

def _show_version(value: Optional[bool]) -> None:  # pragma: no cover
    if value:
        rprint(f"[cyan]qctddft[/] version [bold]{_VERSION}[/]")
        raise typer.Exit()

# ---------------- extract ----------------
@app.command("extract")
def extract_cmd(
    input_glob: str = typer.Argument(..., help="Glob for Q-Chem TDDFT outputs (quote it!)."),
    out_csv: str = typer.Option("es_extracted.dat", "--out", "-o", help="Output CSV path."),
    max_states: int = typer.Option(5, help="Max excited states to capture per snapshot."),
    f1_min: float = typer.Option(0.1, help="First-state oscillator strength threshold (keep if f1≥this)."),
    env: str = typer.Option("auto", help="Environment: efp | pcm | vacuum | auto (default: auto)."),
    strict_env: bool = typer.Option(False, help="If True, skip files that don't match the requested env."),
    nprocs: int = typer.Option(1, help="Parallel processes for parsing (1 = serial)."),
    config_min: Optional[int] = typer.Option(None, help="Optional: min configuration # for exclusion report."),
    config_max: Optional[int] = typer.Option(None, help="Optional: max configuration # for exclusion report."),
    excluded_out: Optional[str] = typer.Option(None, help="Optional file to write excluded config IDs."),
):
    env = env.lower().strip()
    if env not in ("efp", "pcm", "vacuum", "auto"):
        rprint("[red]Error:[/] --env must be efp | pcm | vacuum | auto")
        raise typer.Exit(code=2)

    try:
        df = extract_directory(
            input_glob=input_glob,
            max_states=max_states,
            f1_min=f1_min,
            env=env,          # type: ignore[arg-type]
            strict_env=strict_env,
            nprocs=nprocs,
        )
    except FileNotFoundError as e:
        rprint(f"[red]{e}[/]")
        raise typer.Exit(code=2)

    to_wide_csv(df, out_csv, max_states)

    table = Table(title="TDDFT Extraction Summary", show_header=True, header_style="bold cyan")
    table.add_column("Output"); table.add_column("Rows"); table.add_column("Env")
    table.add_column("Max States"); table.add_column("f1_min"); table.add_column("nprocs")
    table.add_row(out_csv, str(len(df)), env, str(max_states), f"{f1_min:g}", str(nprocs))
    rprint(table)

    if config_min is not None and config_max is not None and config_min <= config_max:
        have = set(int(c) for c in df["Configuration"] if c != -1) if not df.empty else set()
        full = set(range(int(config_min), int(config_max) + 1))
        missing = sorted(full - have)
        rprint(f"[cyan]Excluded configs in [{config_min}, {config_max}]:[/] {len(missing)}")
        if excluded_out:
            try:
                with open(excluded_out, "w") as fh:
                    for m in missing:
                        fh.write(f"so3sq_{m}\n")
                rprint(f"[green]Wrote[/] excluded list → {excluded_out}")
            except OSError as ex:
                rprint(f"[red]Failed to write excluded list:[/] {ex}")

    raise typer.Exit(code=0)

# ---------------- regions ----------------
@app.command("regions")
def regions_cmd(
    spectrum_csv: str = typer.Argument(..., help="CSV with columns: Energy (eV), Intensity"),
    n_regions: int = typer.Option(3, help="How many regions to retain."),
    prom_q: float = typer.Option(0.60, help="Prominence quantile for d² maxima."),
    min_width_e: float = typer.Option(0.035, help="Minimum region width (eV) before merging slivers."),
    selection: str = typer.Option("top", help="Selection strategy: top | centered"),
    y_floor_frac: float = typer.Option(0.0, help="Subtract this fraction of max intensity (baseline)."),
    plot: bool = typer.Option(False, help="Show plots (spectrum + d²I/dE²)."),
):
    df = pd.read_csv(spectrum_csv)
    cols = [c.lower() for c in df.columns]
    try:
        e = df[df.columns[cols.index("energy")]].to_numpy()
        y = df[df.columns[cols.index("intensity")]].to_numpy()
    except ValueError:
        rprint("[red]Error:[/] spectrum CSV must have 'Energy' and 'Intensity' columns")
        raise typer.Exit(code=2)

    sel = selection.lower().strip()
    if sel not in ("top", "centered"):
        rprint("[red]Error:[/] --selection must be 'top' or 'centered'")
        raise typer.Exit(code=2)

    regs, arrays = identify_regions(
        e, y,
        n_regions=n_regions,
        prominence_quantile=prom_q,
        min_width_e=min_width_e,
        selection=sel,     # type: ignore[arg-type]
        y_floor_frac=y_floor_frac,
    )
    rprint("[cyan]Regions:[/]")
    for k, r in enumerate(regs, 1):
        rprint(f"  {k}: {r.left_energy:.4f} — {r.right_energy:.4f}  | "
               f"peak @{r.peak_energy:.4f} (I={r.peak_intensity:.4g})")

    if plot:
        plot_regions(e, y, regs, arrays, show_d2=True, show_only_used_boundaries=True)

def run():  # console entry
    app()

if __name__ == "__main__":
    run()
