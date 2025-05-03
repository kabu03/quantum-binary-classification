# Quantum vs Classical SVM for Binary Classification

This repository contains the Python implementation for a comparative study between classical Support Vector Machines (SVM) and Quantum Support Vector Machines (QSVM) for binary classification tasks.

The comparison utilizes Qiskit for the quantum implementation and Scikit-learn for the classical baseline, tested on the [Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication) and synthetic Two Moons datasets. The primary goal is to evaluate performance (accuracy, F1-score) and resource usage (training/prediction time) across different classical and quantum (simulated/hardware) backends.

## Directory Structure

*   `data/`: Contains datasets (e.g., `data_banknote_authentication.txt`).
*   `results/`: Default directory where experiment results (metrics, timings, configs) are saved in timestamped subdirectories (e.g., `run_YYYYMMDD_HHMMSS/`).
*   `scripts/`: Utility scripts, such as `generate_latex_table.py` for processing results.
*   `src/`: Main source code.
    *   `compare.py`: **Main entry point** to run the comparison experiments.
    *   `classical_svm.py`: Implements the classical SVM pipeline.
    *   `quantum_svm.py`: Implements the QSVM pipeline using Qiskit.
    *   `data_utils.py`: Handles data loading, cleaning, splitting, and scaling.
    *   `metrics_logger.py`: Utility for saving experiment results to files.
*   `requirements.txt`: Lists Python package dependencies.
*   `.env.example`: Template for environment variables (e.g., IBM Quantum token).
*   `README.md`: This file.

## Quick Start

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up IBM Quantum Token (Optional, for real hardware):**
    *   To run experiments on actual IBM Quantum hardware (e.g., `ibm_sherbrooke`), you need to authenticate using your API token.  
    
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your IBM Quantum API token:
        ```plaintext
        # .env
        IBMQ_TOKEN=YOUR_TOKEN_HERE
        ```
        *(Note: Ensure `.env` is listed in your `.gitignore` file to avoid committing your actual token.)*

5.  **Run the comparison:**
    Execute the main comparison script **from the project root directory**. Results will be saved in the `results/` directory.
    ```bash
    python -m src.compare [OPTIONS]
    ```

## Usage (`src/compare.py`)

The main script `src/compare.py` orchestrates the experiments. Key command-line arguments allow you to control the execution:

*   `--banknote_samples <N>`: Use `<N>` samples from the Banknote dataset (default: 200). Capped by `MAX_BANKNOTE_SAMPLES`.
*   `--two_moons_samples <N>`: Use `<N>` samples from the Two Moons dataset (default: 200). Capped by `MAX_TWO_MOONS_SAMPLES`.
*   `--skip_q_hparam`: If included, bypasses the potentially time-consuming quantum hyperparameter search and uses default parameters for QSVM. Recommended for quick tests or when using real hardware with limited time.

**Examples (run from project root):**

```bash
# Run with 300 banknote, 500 two_moons samples, skip quantum hparam search
python -m src.compare --banknote_samples 300 --two_moons_samples 500 --skip_q_hparam

# Run with 100 samples each, INCLUDING quantum hparam search (can be slow!)
python -m src.compare --banknote_samples 100 --two_moons_samples 100

# Run with default settings (200 samples each, includes quantum hparam search)
python -m src.compare
```

## Output
Each run creates a unique subdirectory within `results` (e.g., `results/run_20250503_100000/`). This directory contains:

- `run_config.txt`: The configuration used for the run (sample sizes, skip flag).  
- `metrics_*.json`: JSON files containing detailed evaluation metrics (accuracy, F1, etc.) for each backend/dataset combination.  
- `timings.csv`: A CSV file summarizing the training and prediction times for each experiment.  

The script also prints a summary table of results to the console upon completion.

## Generating LaTeX Tables (Optional)
A utility script is provided to generate a LaTeX table summarizing the results from a specific run directory. This is useful for incorporating results into research papers or reports.

**Usage:**

Run the script from the project root directory, providing the path to the desired run directory as an argument:  
```bash
python scripts/generate_latex_table.py results/run_YYYYMMDD_HHMMSS
```

Replace `results/run_YYYYMMDD_HHMMSS` with the actual path to your results folder. The script will print the LaTeX code for the table to the console. You will need the booktabs LaTeX package `(\usepackage{booktabs})` for the table formatting.

## Dependencies
All required Python packages are listed in `requirements.txt.`

## Contributing
Contributions, issues, and pull requests are welcome! If you encounter any problems or have suggestions for improvements, please open an issue on the repository's issue tracker.

