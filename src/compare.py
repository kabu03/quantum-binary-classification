"""
compare.py  –  orchestrates *all* experiments and prints a compact results table.

Usage
------
python -m src.compare                   # default backends
BACKEND=qasm python -m src.compare      # override quantum backend
"""

import json, csv, os, shutil, importlib, textwrap
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------------- config
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), "..", "results")
CLASSICAL_MODULE = "src.classical_svm"
QUANTUM_MODULE   = "src.quantum_svm"

DATASETS         = ("banknote", "two_moons")
# ------------------------------------------------------------------- helpers
def fresh_results_dir():
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def pretty_table(rows, headers):
    col_w = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_w)
    line = "-+-".join("-" * w for w in col_w)
    print(fmt.format(*headers))
    print(line)
    for r in rows:
        print(fmt.format(*r))

def load_metrics():
    metrics = []   # list of dicts
    print(f"[DEBUG load_metrics] Looking in directory: {RESULTS_DIR}") # Add print
    if not os.path.exists(RESULTS_DIR):
        print(f"[DEBUG load_metrics] Directory NOT FOUND: {RESULTS_DIR}")
        return metrics
    
    files_found = os.listdir(RESULTS_DIR)
    print(f"[DEBUG load_metrics] Files found: {files_found}") # Add print

    for fn in files_found:
        if fn.startswith("metrics_") and fn.endswith(".json"):
            file_path = os.path.join(RESULTS_DIR, fn)
            print(f"[DEBUG load_metrics] Attempting to load: {file_path}") # Add print
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    metrics.append(data | {"file": fn})
                    print(f"[DEBUG load_metrics] Successfully loaded {fn}") # Add print
            except Exception as e:
                print(f"[DEBUG load_metrics] FAILED to load {fn}: {e}") # Add print for errors
        # else: # Optional: See skipped files
        #     print(f"[DEBUG load_metrics] Skipping file: {fn}")
    return metrics

def load_timings():
    timings = {}
    path = os.path.join(RESULTS_DIR, "timings.csv")
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
    # 1) clean slate
    fresh_results_dir()

    # 2) import modules dynamically
    cmod = importlib.import_module(CLASSICAL_MODULE)
    qmod = importlib.import_module(QUANTUM_MODULE)

    # 3) decide quantum backend from env
    backend_env = os.getenv("BACKEND", "statevector")   # e.g. qasm, ['ibm_brisbane', 'ibm_sherbrooke']
    quantum_backend_name = f"qsvm_{backend_env}"

    # 4) run experiments
    print(f"[RUN] Classical SVMs …")
    cmod.run_banknote()
    cmod.run_two_moons()

    print(f"[RUN] Quantum SVMs on backend={backend_env} …")
    qmod.run_banknote(backend_name=quantum_backend_name, skip_hparam_search=True)
    qmod.run_two_moons(backend_name=quantum_backend_name, skip_hparam_search=True)

    # 5) read results and print table
    print("[DEBUG main] Loading metrics...") # Add print
    metrics = load_metrics()
    print(f"[DEBUG main] Metrics loaded ({len(metrics)} records): {metrics}") # Add print
    
    print("[DEBUG main] Loading timings...") # Add print
    timings = load_timings()
    print(f"[DEBUG main] Timings loaded: {timings}") # Add print


    rows   = []
    print("[DEBUG main] Starting results table loop...") # Add print
    for dset in DATASETS:
        for variant in ("classical_svm_cpu", quantum_backend_name):
            print(f"[DEBUG main] Looking for: dataset='{dset}', backend='{variant}'") # Add print
            try:
                # --- Keep using next() for now to reproduce the error ---
                record = next(m for m in metrics
                              if m["dataset"] == dset and m["backend"] == variant)
                print(f"[DEBUG main] Found record: {record}") # Add print
                # ---
                tkey   = (variant, dset)
                tinfo  = timings.get(tkey, {})
                rows.append([
                    dset,
                    variant,
                    f"{record['accuracy']:.3f}",
                    f"{record['f1_score']:.3f}",
                    tinfo.get("train_time_seconds", "--"),
                    tinfo.get("predict_time_seconds", "--"),
                ])
            except StopIteration:
                 print(f"[ERROR main] StopIteration: Record NOT FOUND for dataset='{dset}', backend='{variant}'") # Add specific error print
                 # Optional: Re-raise the error if you want the script to stop here
                 # raise
                 # Optional: Or append a placeholder row
                 rows.append([dset, variant, "ERR", "ERR", "--", "--"])
            except Exception as e:
                 print(f"[ERROR main] Unexpected error for dataset='{dset}', backend='{variant}': {e}") # Catch other potential errors
                 rows.append([dset, variant, "ERR", "ERR", "--", "--"])


    print()
    pretty_table(rows,
        ["Dataset", "Backend", "Acc", "F1", "Train-s", "Pred-s"])

if __name__ == "__main__":
    main()
