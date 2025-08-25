from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict
from . import logger

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
except ImportError:
    logger.error("MDAnalysis, scikit-learn, and matplotlib must be installed. Run 'pip install MDAnalysis scikit-learn matplotlib'")
    raise

def get_pdb_paths(pdb_glob: str, config_ids: List[int]) -> Dict[int, str]:
    """Finds and maps PDB file paths to their configuration IDs."""
    pdb_paths = {}
    all_files = glob.glob(pdb_glob)
    for cfg_id in config_ids:
        # A flexible search to find the PDB file for a given config ID
        found = False
        for f in all_files:
            if f"_{cfg_id}.pdb" in f or f"_{cfg_id}_" in f:
                pdb_paths[cfg_id] = f
                found = True
                break
        if not found:
            logger.warning(f"Could not find a PDB file for configuration {cfg_id}")
    return pdb_paths

def _load_coordinates(pdb_paths: Dict[int, str], solute_selection: str) -> Tuple[np.ndarray, List[int]]:
    """Loads and flattens solute coordinates from PDB files."""
    coordinates = []
    loaded_configs = []
    logger.info("Loading atomic coordinates for the solute selection: '%s'", solute_selection)
    for cfg_id, path in pdb_paths.items():
        try:
            universe = mda.Universe(path)
            solute = universe.select_atoms(solute_selection)
            if solute.n_atoms == 0:
                logger.warning(f"No atoms selected for '{solute_selection}' in {path}. Skipping.")
                continue
            coordinates.append(solute.positions.flatten())
            loaded_configs.append(cfg_id)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
    return np.array(coordinates), loaded_configs

def analyze_k_value(X: np.ndarray, max_k: int, output_dir: Path):
    """Performs Elbow and Silhouette analysis to find the optimal K."""
    logger.info(f"Analyzing optimal K from 2 to {max_k}...")
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    # Plot Elbow Method
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.title('Elbow Method for Optimal K')
    elbow_path = output_dir / "elbow_plot.png"
    plt.savefig(elbow_path)
    logger.info(f"Saved Elbow plot to {elbow_path}")
    plt.close()

    # Plot Silhouette Scores
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal K')
    silhouette_path = output_dir / "silhouette_plot.png"
    plt.savefig(silhouette_path)
    logger.info(f"Saved Silhouette plot to {silhouette_path}")
    plt.close()
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    logger.info(f"Optimal K based on Silhouette Score: {optimal_k}")


def cluster_structures(
    assignment_csv: str,
    pdb_glob: str,
    region_id: int,
    n_clusters: int = 5,
    solute_selection: str = "resname ZN",
    output_dir: str = "clusters",
    method: str = "kmeans",
    analyze_k: bool = False,
    max_k: int = 15,
):
    """
    Performs clustering on PDB structures belonging to a specific spectral region.
    """
    logger.info("Loading assignments and filtering for region %d", region_id)
    assign_df = pd.read_csv(assignment_csv)
    region_df = assign_df[assign_df["region"] == region_id]
    if region_df.empty:
        raise ValueError(f"No assignments found for region {region_id}.")

    config_ids = sorted(region_df["Configuration"].unique())
    logger.info(f"Found {len(config_ids)} unique configurations for region {region_id}")

    pdb_paths = get_pdb_paths(pdb_glob, config_ids)
    if not pdb_paths:
        raise FileNotFoundError("Could not find any PDB files for the given configurations.")

    X, loaded_configs = _load_coordinates(pdb_paths, solute_selection)
    if len(loaded_configs) < n_clusters:
        raise ValueError(f"Number of loaded structures ({len(loaded_configs)}) is less than n_clusters ({n_clusters}).")

    output_path_obj = Path(output_dir)
    output_path_obj.mkdir(exist_ok=True)

    if analyze_k:
        analyze_k_value(X, max_k, output_path_obj)
        return

    logger.info(f"Performing {method} clustering to find {n_clusters} clusters...")
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError("Method must be 'kmeans' or 'hierarchical'")
        
    labels = model.fit_predict(X)

    logger.info("Identifying and saving representative structures for each cluster.")
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            logger.warning(f"Cluster {i+1} is empty. Skipping.")
            continue
        
        # Find the structure closest to the center (centroid)
        if method == "kmeans":
            cluster_center = model.cluster_centers_[i]
            distances = np.linalg.norm(X[cluster_indices] - cluster_center, axis=1)
        else: # For hierarchical, find the medoid (most central point)
            cluster_points = X[cluster_indices]
            distances = np.sum(np.linalg.norm(cluster_points - point, axis=1) for point in cluster_points)

        centroid_index_in_cluster = np.argmin(distances)
        centroid_config_id = loaded_configs[cluster_indices[centroid_index_in_cluster]]
        
        # Save the representative PDB file
        out_pdb_path = output_path_obj / f"region_{region_id}_cluster_{i+1}_config_{centroid_config_id}.pdb"
        u = mda.Universe(pdb_paths[centroid_config_id])
        u.atoms.write(out_pdb_path)
        logger.info(f"Saved representative for cluster {i+1} -> {out_pdb_path.name}")