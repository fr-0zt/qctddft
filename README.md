# qctddft: A Tool for Q-Chem TDDFT Analysis

This package provides a suite of tools for processing and analyzing data from Q-Chem Time-Dependent Density Functional Theory (TDDFT) calculations.

## Features

-   **Extract** TDDFT data from Q-Chem output files.
-   **Generate** convoluted, broadened absorption spectra.
-   **Analyze** spectra to identify and characterize distinct regions.
-   **Assign** individual states to these spectral regions.

## Installation

\`\`\`bash
pip install .
\`\`\`

## Usage

The package provides a command-line interface with three main commands:

### 1. `extract`

Extracts TDDFT data from a directory of Q-Chem output files.

\`\`\`bash
qctddft extract "path/to/*.out" --out extracted_data.csv
\`\`\`

### 2. `spectrum`

Generates a convoluted spectrum from the extracted data.

\`\`\`bash
qctddft spectrum extracted_data.csv --out spectrum.csv --save-plot
\`\`\`

### 3. `regions`

Analyzes a spectrum to identify and plot spectral regions.

\`\`\`bash
qctddft regions spectrum.csv --n-regions 3 --save
\`\`\`
