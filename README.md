# qctddft: A Toolkit for TDDFT Analysis

`qctddft` is a command-line toolkit designed to streamline the post-processing of Time-Dependent Density Functional Theory (TDDFT) calculations from Q-Chem. It provides an end-to-end workflow, from raw output files to insightful structural analysis, enabling a direct link between calculated spectral features and the underlying molecular geometries.

## Features

  - **Extract**: Efficiently parse Q-Chem output files to extract TDDFT excitation energies and oscillator strengths.
  - **Spectrum**: Generate publication-quality, broadened absorption spectra using Gaussian convolution.
  - **Regions**: Automatically identify and partition spectra into distinct regions (peaks and shoulders) using a second-derivative analysis.
  - **Assign**: Attribute individual excited states to the identified spectral regions using both hard and fractional (Gaussian overlap) assignment methods.
  - **Cluster**: Perform K-Means or Hierarchical clustering on the molecular structures within a spectral region to identify key representative geometries.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/fr-0zt/qctddft.git
    cd qctddft
    ```

2.  **Install the package:**

    ```bash
    pip install .
    ```

## Workflow and Usage

The `qctddft` toolkit is designed to be used as a sequential pipeline. Below is a complete workflow with example commands.

### Step 1: `extract`

Parse a directory of Q-Chem `.out` files to extract the TDDFT data into a single CSV file.

```bash
qctddft extract "path/to/outputs/*.out" --out extracted_data.csv
```

### Step 2: `spectrum`

Generate a convoluted spectrum from the extracted data and save a plot.

```bash
qctddft spectrum extracted_data.csv --out spectrum.csv --save-plot
```

### Step 3: `regions`

Analyze the generated spectrum to identify 3 distinct regions and save the results to a CSV file.

```bash
qctddft regions spectrum.csv --n-regions 3 --out-csv regions.csv --save
```

### Step 4: `assign`

Assign the excited states from the extracted data to the regions you just identified, using a fractional assignment method.

```bash
qctddft assign extracted_data.csv regions.csv --out-prefix assignment_results --fractional
```

This will produce two files: `assignment_results_state_assignment.csv` and `assignment_results_region_summary.csv`.

### Step 5: `cluster`

Find the 5 most representative structures for spectral region 1 using the PDB files from your calculations.

First, you can run the K-analysis to determine the optimal number of clusters:

```bash
qctddft cluster assignment_results_state_assignment.csv "path/to/pdbs/*.pdb" --region 1 --analyze-k
```

Then, perform the clustering with your chosen number of clusters:

```bash
qctddft cluster assignment_results_state_assignment.csv "path/to/pdbs/*.pdb" --region 1 --n-clusters 5 --select "resname UNK" --out-dir representative_structures
```

The representative PDB structures will be saved in the `representative_structures` directory.

## Contributing

Contributions are welcome\! If you have suggestions for improvements or find a bug, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.