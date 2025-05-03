import json, csv, os, shutil, importlib, textwrap, argparse
from datetime import datetime # Import datetime
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------------- config
BASE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results") # Base directory
CLASSICAL_MODULE = "src.classical_svm"
QUANTUM_MODULE   = "src.quantum_svm"

DATASETS         = ("banknote", "two_moons")
# Define the quantum backends to test here
QUANTUM_BACKENDS_TO_TEST = ["ibm_sherbrooke"] # Options are "statevector", "qasm", "ibm_sherbrooke", "ibm_brisbane"

MAX_BANKNOTE_SAMPLES = 1372 # Max samples in the banknote dataset
MAX_TWO_MOONS_SAMPLES = 1500 # Max samples in the two_moons dataset
# ------------------------------------------------------------------- helpers

def pretty_table(rows, headers):
    col_w = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_w)
    line = "-+-".join("-" * w for w in col_w)
    print(fmt.format(*headers))
    print(line)
    for r in rows:
        print(fmt.format(*r))

# Modify load_metrics and load_timings to accept the run_dir
def load_metrics(run_dir):
    metrics = []   # list of dicts
    print(f"[DEBUG load_metrics] Looking in directory: {run_dir}")
    if not os.path.exists(run_dir):
        print(f"[DEBUG load_metrics] Directory NOT FOUND: {run_dir}")
        return metrics

    files_found = os.listdir(run_dir)
    print(f"[DEBUG load_metrics] Files found: {files_found}")

    for fn in files_found:
        if fn.startswith("metrics_") and fn.endswith(".json"):
            file_path = os.path.join(run_dir, fn) # Use run_dir
            print(f"[DEBUG load_metrics] Attempting to load: {file_path}")
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    metrics.append(data | {"file": fn})
                    print(f"[DEBUG load_metrics] Successfully loaded {fn}")
            except Exception as e:
                print(f"[DEBUG load_metrics] FAILED to load {fn}: {e}")
    return metrics

def load_timings(run_dir):
    timings = {}
    path = os.path.join(run_dir, "timings.csv") # Use run_dir
    if not os.path.isfile(path):
        return timings
    with open(path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            key = (row["backend"], row["dataset"])
            timings[key] = row
    return timings

# ------------------------------------------------------------------- main
def main():
    # --- Argument Parsing for Sample Sizes ---
    parser = argparse.ArgumentParser(description="Compare Classical and Quantum SVMs.")
    parser.add_argument('--banknote_samples', type=int, default=200,
                        help=f'Number of samples for banknote dataset (max: {MAX_BANKNOTE_SAMPLES}).')
    parser.add_argument('--two_moons_samples', type=int, default=200,
                        help=f'Number of samples for two_moons dataset (max: {MAX_TWO_MOONS_SAMPLES}).')
    parser.add_argument('--skip_q_hparam', action='store_true', # Add flag to skip quantum hparam search
                        help='Skip quantum hyperparameter search and use defaults.')

    args = parser.parse_args()

    # --- Validate Sample Sizes ---
    banknote_n = min(args.banknote_samples, MAX_BANKNOTE_SAMPLES)
    two_moons_n = min(args.two_moons_samples, MAX_TWO_MOONS_SAMPLES)
    print(f"[INFO] Using {banknote_n} samples for banknote dataset.")
    print(f"[INFO] Using {two_moons_n} samples for two_moons dataset.")
    print(f"[INFO] Skip Quantum HParam Search: {args.skip_q_hparam}")

    # --- Create Unique Run Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_dir = os.path.join(BASE_RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(current_run_dir, exist_ok=True)
    print(f"[SETUP] Results for this run will be saved in: {current_run_dir}")

    # --- Save Run Configuration (Sample Sizes) ---
    config_file_path = os.path.join(current_run_dir, "run_config.txt")
    print(f"[SETUP] Saving run configuration to: {config_file_path}")
    with open(config_file_path, "w") as f:
        f.write(f"Banknote Samples: {banknote_n}\n")
        f.write(f"Two Moons Samples: {two_moons_n}\n")
        f.write(f"Skip Quantum HParam Search: {args.skip_q_hparam}\n")
    # --- End Save Run Configuration ---

    # 2) import modules dynamically
    print("[SETUP] Importing modules...")
    cmod = importlib.import_module(CLASSICAL_MODULE)
    qmod = importlib.import_module(QUANTUM_MODULE)

    # 3) Run Classical Experiments (passing sample size AND run_dir)
    print(f"[RUN] Classical SVMs ({banknote_n} banknote, {two_moons_n} two_moons samples)...")
    cmod.run_banknote(num_samples=banknote_n, run_dir=current_run_dir) # Pass run_dir
    cmod.run_two_moons(num_samples=two_moons_n, run_dir=current_run_dir) # Pass run_dir

    # 4) Loop through Quantum Backends (passing sample size, skip flag, AND run_dir)
    print(f"[RUN] Quantum SVMs ({banknote_n} banknote, {two_moons_n} two_moons samples)...")
    # --- Keep track of the constructed names for results table ---
    all_quantum_log_names = []
    for backend_type in QUANTUM_BACKENDS_TO_TEST:
        # Construct the name used for logging here as well, for the results table
        log_backend_name = f"qsvm_{backend_type}"
        all_quantum_log_names.append(log_backend_name)
        print(f"\n--- Running Quantum Backend: {backend_type} ---")

        try:
            # --- Pass only backend_type ---
            qmod.run_banknote(backend_type=backend_type,
                              num_samples=banknote_n,
                              skip_hparam_search=args.skip_q_hparam,
                              run_dir=current_run_dir)
            qmod.run_two_moons(backend_type=backend_type,
                               num_samples=two_moons_n,
                               skip_hparam_search=args.skip_q_hparam,
                               run_dir=current_run_dir)
        except Exception as e:
             print(f"[ERROR] Failed running backend type '{backend_type}': {e}")

    # 5) read results and print table (load from current_run_dir)
    print("\n[RESULTS] Consolidating results for this run...")
    metrics = load_metrics(current_run_dir) # Pass run_dir
    timings = load_timings(current_run_dir) # Pass run_dir

    rows   = []
    # --- Use the constructed log names for the results table ---
    all_variants = ["classical_svm_cpu"] + all_quantum_log_names
    print(f"[DEBUG main] Expected variants: {all_variants}")

    for dset in DATASETS:
        for variant in all_variants: # variant is now e.g., "classical_svm_cpu", "qsvm_statevector"
            print(f"[DEBUG main] Looking for: dataset='{dset}', backend='{variant}'")
            try:
                record = next(m for m in metrics
                              if m["dataset"] == dset and m["backend"] == variant)
                # --- Use variant directly as the key for timings ---
                tkey   = (variant, dset)
                tinfo  = timings.get(tkey, {})
                rows.append([
                    dset,
                    variant, # Use variant directly
                    f"{record['accuracy']:.3f}",
                    f"{record['f1_score']:.3f}",
                    tinfo.get("train_time_seconds", "--"),
                    tinfo.get("predict_time_seconds", "--"),
                ])
            except StopIteration:
                 print(f"[WARN main] StopIteration: Record NOT FOUND for dataset='{dset}', backend='{variant}'") # Changed to WARN
                 # Add placeholder row only if you want to see ERR in the table for missing runs
                 rows.append([dset, variant, "MISS", "MISS", "--", "--"])
            except Exception as e:
                 print(f"[ERROR main] Unexpected error processing results for dataset='{dset}', backend='{variant}': {e}")
                 rows.append([dset, variant, "ERR", "ERR", "--", "--"])

    print("\n--- Final Results ---")
    pretty_table(rows,
        ["Dataset", "Backend", "Acc", "F1", "Train-s", "Pred-s"])

if __name__ == "__main__":
    main()