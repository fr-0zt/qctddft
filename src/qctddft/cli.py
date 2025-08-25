from __future__ import annotations
import sys
import typer
import pandas as pd
from typing import Optional
from pathlib import Path
from . import logger, __version__
from .extract import extract_directory
from .spectra import build_normalized_spectrum, write_spectrum_csv
from .regions import identify_regions
from .plots import plot_regions
from .io import write_regions_csv, read_tddft_table, read_regions_csv
from .assign import assign_states_to_regions
from .cluster import cluster_structures

App = typer.Typer(add_completion=False, no_args_is_help=True)

env_opt = typer.Option(
    "auto",
    help="Environment: 'efp' (use polarization-corrected pass), 'pcm'/'vac' (first pass), or 'auto'.",
)

def _env_validator(v: str) -> str:
    v = (v or "auto").lower()
    if v not in {"auto", "efp", "pcm", "vac"}:
        raise typer.BadParameter("env must be one of: auto, efp, pcm, vac")
    return v

@App.callback()
def _root(
    version: Optional[bool] = typer.Option(None, "--version", help="Show version and exit.", is_eager=True),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase logging verbosity"),
):
    if version:
        typer.echo(f"qctddft {__version__}")
        raise typer.Exit()
    if verbose >= 2:
        logger.setLevel("DEBUG")
    elif verbose == 1:
        logger.setLevel("INFO")
    else:
        logger.setLevel("WARNING")

# ------------------ extract ------------------
@App.command("extract")
def extract_cmd(
    input_glob: str = typer.Argument(..., help="Glob of Q-Chem TDDFT output files (*.out)"),
    out_csv: str = typer.Option("extracted.csv", "--out", help="Output CSV path"),
    max_states: int = typer.Option(5, help="Max number of states to extract per snapshot"),
    f1_min: float = typer.Option(0.1, help="First-state inclusion threshold on oscillator strength"),
    env: str = env_opt,
    config_min: Optional[int] = typer.Option(None, help="Keep configs ≥ this ID"),
    config_max: Optional[int] = typer.Option(None, help="Keep configs ≤ this ID"),
    strict_env: bool = typer.Option(False, help="If true, do not auto-disambiguate EFP vs non-EFP."),
):
    """
    Extract TDDFT states from Q-Chem outputs.
    Columns: Configuration, Energy 1..N, Strength 1..N
    """
    logger.info("Scanning: %s", input_glob)
    df = extract_directory(
        input_glob=input_glob,
        max_states=max_states,
        f1_min=f1_min,
        env=env,
        config_min=config_min,
        config_max=config_max,
        strict_env=strict_env,
    )
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s  (rows: %d)", out_csv, len(df))

# ------------------ spectrum ------------------
@App.command("spectrum")
def spectrum_cmd(
    extracted_csv: str = typer.Argument(..., help="CSV from 'extract' with Energy*/Strength*"),
    output_csv: str = typer.Option(None, "--out", help="Output path for the spectrum CSV file."),
    sigma: float = typer.Option(0.04, "--sigma", help="Gaussian FWHM in eV."),
    emin: float = typer.Option(1.7, "--emin"),
    emax: float = typer.Option(2.4, "--emax"),
    npts: int = typer.Option(1000, "--npts"),
    fmin_snapshot: float = typer.Option(0.10, "--fmin-snapshot", help="Filter by FIRST state (E1>0 & f1>=)"),
    fmin_state: float = typer.Option(0.0, "--fmin-state", help="Per-state f filter after snapshot filter"),
    states: str = typer.Option("all", "--states", help="Use 'all' states or 'first' only"),
    save_plot: bool = typer.Option(False, "--save-plot", help="Generate and save a plot of the spectrum."),
):
    """
    Generates a convoluted spectrum from the extracted TDDFT data.
    """
    out_path = output_csv or (str(Path(extracted_csv).with_suffix("")) + "_spectrum-norm.csv")

    x, y = build_normalized_spectrum(
        extracted_csv,
        sigma=sigma, emin=emin, emax=emax, npts=npts,
        fmin_snapshot=fmin_snapshot, fmin_state=fmin_state, states=states,
        save_plot=save_plot,
        output_path=out_path,
    )
    write_spectrum_csv(out_path, x, y)
    typer.echo(f"[OK] Wrote spectrum to {out_path}")

# ------------------ regions ------------------
@App.command("regions")
def regions_cmd(
    spectrum_csv: str = typer.Argument(..., help='CSV with "Energy (eV),Intensity"'),
    out_csv: Optional[str] = typer.Option(None, "--out-csv", help="Output CSV path for identified regions."),
    n_regions: int = typer.Option(3, "--n_regions"),
    prom_q: float = typer.Option(0.60, "--prom_q"),
    min_width_e: float = typer.Option(0.02, "--min_width_e"),
    selection: str = typer.Option("top", "--select", help="top|centered"),
    y_floor_frac: float = typer.Option(0.00, "--y_floor_frac"),
    band_area_q: float = typer.Option(0.00, "--band_area_q"),
    save_plots: bool = typer.Option(False, "--save"),
):
    """
    Identifies and analyzes regions in a spectrum.
    """
    df = pd.read_csv(spectrum_csv)
    cols = [c for c in df.columns if df[c].dtype.kind in "fc"]
    if len(cols) < 2:
        raise typer.BadParameter("Spectrum CSV must have at least two numeric columns (energy, intensity).")
    e = df[cols[0]].to_numpy(float)
    i = df[cols[1]].to_numpy(float)

    regs, arr = identify_regions(
        e, i,
        n_regions=n_regions,
        prom_q=prom_q,
        min_width_e=min_width_e,
        selection=selection,
        y_floor_frac=y_floor_frac,
        band_area_q=band_area_q,
    )
    
    typer.echo("Identified Regions:")
    table_df = pd.DataFrame([vars(r) for r in regs])
    typer.echo(table_df[['left_energy', 'right_energy', 'peak_energy', 'peak_intensity']].round(4).to_string(index=False))
    
    if out_csv:
        write_regions_csv(out_csv, regs)

    if save_plots:
        plot_regions(e, i, regs, arr, show_d2=True,
                     save_path_spectrum=(str(Path(spectrum_csv).with_suffix(""))+"_regions.png"),
                     save_path_d2=(str(Path(spectrum_csv).with_suffix(""))+"_d2_edges.png"))

# ------------------ assign ------------------
@App.command("assign")
def assign_cmd(
    extracted_csv: str = typer.Argument(..., help="CSV file from the 'extract' command."),
    regions_csv: str = typer.Argument(..., help="CSV file from the 'regions' command."),
    out_prefix: str = typer.Option("assigned", "--out-prefix", help="Prefix for the output assignment and summary files."),
    fractional: bool = typer.Option(False, "--fractional", help="Use fractional (Gaussian overlap) assignment instead of hard assignment."),
    sigma: float = typer.Option(0.04, "--sigma", help="Gaussian FWHM (eV) for fractional assignment."),
    states: str = typer.Option("all", "--states", help="Use 'all' states or 'first' only for assignment."),
    f_cutoff: float = typer.Option(0.1, "--f-cutoff", help="Minimum oscillator strength (f) for a state to be included in assignment."), # <-- New option
):
    """
    Assigns excited states from snapshots to the identified spectral regions.
    """
    typer.echo(f"Reading extracted data from {extracted_csv}")
    df, e_cols, f_cols, cfg_col = read_tddft_table(extracted_csv)
    
    typer.echo(f"Reading region definitions from {regions_csv}")
    regions = read_regions_csv(regions_csv)

    typer.echo(f"Assigning states to {len(regions)} regions (f >= {f_cutoff})...")
    assign_states_to_regions(
        df=df,
        energy_cols=e_cols,
        strength_cols=f_cols,
        config_col=cfg_col,
        regions=regions,
        sigma=sigma,
        states_mode=states,
        fractional=fractional,
        f_cutoff=f_cutoff, # <-- Pass the new value
        save=True,
        table_path=out_prefix,
    )
    typer.secho(f"✓ Assignment complete. Output files are prefixed with '{out_prefix}'.", fg=typer.colors.GREEN)

# ------------------ cluster ------------------
@App.command("cluster")
def cluster_cmd(
    assignment_csv: str = typer.Argument(..., help="CSV file of state assignments."),
    pdb_glob: str = typer.Argument(..., help="Glob pattern for PDB files (e.g., 'path/to/*.pdb')."),
    region_id: int = typer.Option(..., "--region", help="The region ID to analyze."),
    n_clusters: int = typer.Option(5, "--n-clusters", help="Number of clusters to find."),
    solute_selection: str = typer.Option("resname ZN", "--select", help="MDAnalysis selection string for the solute."),
    output_dir: str = typer.Option("clusters", "--out-dir", help="Directory to save output files."),
    method: str = typer.Option("kmeans", "--method", help="Clustering algorithm: 'kmeans' or 'hierarchical'."),
    analyze_k: bool = typer.Option(False, "--analyze-k", help="Perform analysis to find the optimal K and exit."),
    max_k: int = typer.Option(15, "--max-k", help="Maximum K to test for the analysis."),
):
    """
    Cluster structures in a spectral region to find representative geometries.
    """
    try:
        cluster_structures(
            assignment_csv=assignment_csv,
            pdb_glob=pdb_glob,
            region_id=region_id,
            n_clusters=n_clusters,
            solute_selection=solute_selection,
            output_dir=output_dir,
            method=method,
            analyze_k=analyze_k,
            max_k=max_k,
        )
        if analyze_k:
            typer.secho(f"✓ K-value analysis complete. Plots saved in '{output_dir}'.", fg=typer.colors.GREEN)
        else:
            typer.secho(f"✓ Clustering complete. Representative structures saved in '{output_dir}'.", fg=typer.colors.GREEN)

    except (ValueError, FileNotFoundError) as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except ImportError as e:
        typer.secho(f"Missing dependency: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

def run():
    try:
        App()
    except KeyboardInterrupt:
        typer.secho("Interrupted.", fg=typer.colors.YELLOW)
        sys.exit(130)