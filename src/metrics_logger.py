import os
import json
import csv

# Modify save functions to accept run_dir
def save_metrics(metrics: dict, backend_name: str, run_dir: str):
    """Saves evaluation metrics to a JSON file within the specified run directory."""
    fn = f"metrics_{backend_name}_{metrics['dataset']}.json"
    path = os.path.join(run_dir, fn) # Use run_dir
    print(f"[SAVE] Saving metrics to: {path}")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def save_timing(train_s: float, predict_s: float, backend: str, dataset: str, run_dir: str):
    """Appends timing info to timings.csv within the specified run directory."""
    path = os.path.join(run_dir, "timings.csv") # Use run_dir
    print(f"[SAVE] Saving timing to: {path}")
    # file exists? write header only if it does not
    hdr = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wrt = csv.writer(f)
        if hdr:
            wrt.writerow(["backend", "dataset", "train_time_seconds", "predict_time_seconds"])
        wrt.writerow([backend, dataset, train_s, predict_s])