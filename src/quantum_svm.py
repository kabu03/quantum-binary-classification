import sys, time, json, psutil, numpy as np
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_utils import load_banknote, generate_two_moons, clean_split_scale
from metrics_logger import save_metrics, save_timing

# -------------- safety helper ---------------------------------
def check_memory(thresh_pct: int = 85):
    """Exit if RAM usage exceeds threshold percentage."""
    if psutil.virtual_memory().percent > thresh_pct:
        print(f"[MEM] RAM usage {psutil.virtual_memory().percent}% > {thresh_pct}% — aborting.")
        sys.exit(1)

def build_feature_map(kind: str, dim: int, reps: int, ent: str):
    if kind == "ZZ":
        return ZZFeatureMap(feature_dimension=dim, reps=reps, entanglement=ent)
    elif kind == "Z":
        return ZFeatureMap(feature_dimension=dim, reps=reps)   # no entanglement arg
    else:
        raise ValueError("Unknown map")

# -------------- hyper-parameter search -------------------------
def search_quantum_hparams(X_train, y_train, X_val, y_val):

    grids = [
        {"map": "ZZ", "reps": 1, "ent": "full"},
        {"map": "ZZ", "reps": 2, "ent": "full"},
        {"map": "Z",  "reps": 2, "ent": "linear"},
    ]
    best_f1, best_cfg, best_C = 0, None, None

    for idx, cfg in enumerate(grids):
        try:
            print(f"\n[INFO] Testing configuration {idx + 1}/{len(grids)}: {cfg}")
            check_memory()

            fmap = build_feature_map(cfg["map"], X_train.shape[1], cfg["reps"], cfg["ent"])
            print("[DEBUG] Feature map created successfully.")

            quantum_kernel = FidelityQuantumKernel(feature_map=fmap, max_circuits_per_job=900)

            print("[DEBUG] Kernel matrices computed successfully.")

            for C in [0.1, 1, 10]:
                svc = QSVC(quantum_kernel=quantum_kernel, C=C)
                svc.fit(X_train, y_train)
                preds = svc.predict(X_val)
                f1 = f1_score(y_val, preds)

                print(f"[TUNE] Config {cfg}, C={C} -> F1={f1:.3f}")
                if f1 > best_f1:
                    best_f1, best_cfg, best_C = f1, cfg, C

        except Exception as e:
            print(f"[ERROR] {cfg}: {e}")

        finally:
            for obj in ("kernel", "svc"):
                if obj in locals():
                    del locals()[obj]
                    import gc; gc.collect()
            print("[DEBUG] Cleaned up memory.")

    print(f"\n[INFO] Best configuration: {best_cfg} with C={best_C} and F1={best_f1:.3f}")
    return best_cfg, best_C

# -------------- training + evaluation --------------------------
def train_quantum(X_train, y_train, fmap, best_C):
    """
    Train QSVC with the chosen feature-map and C.
    """
    quantum_kernel = FidelityQuantumKernel(feature_map=fmap,
                                           max_circuits_per_job=900)

    model = QSVC(quantum_kernel=quantum_kernel, C=best_C, verbose=True)
    model.fit(X_train, y_train)
    return model



def eval_and_log(model, X_test, y_test, dataset_name, backend_name):
    """
    Evaluate the trained model on test data, save results.
    """

    print("[EVAL] Making predictions...", flush=True)
    t0 = time.time()
    y_pred = model.predict(X_test)
    duration = time.time() - t0

    metrics = {
        "dataset": dataset_name,
        "backend": backend_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion": confusion_matrix(y_test, y_pred).tolist()
    }
    print(json.dumps(metrics, indent=2))

    save_metrics(metrics, backend_name)
    save_timing(duration, backend_name, dataset_name)


# -------------- wrappers for each dataset ----------------------
def run_banknote(backend_name="qsvm_statevector"):
    df = load_banknote()
    df = df.sample(n=400, random_state=42)  # Subsample for practical quantum run

    X_train, X_val, X_test, y_train, y_val, y_test = clean_split_scale(df)

    best_cfg, best_C = search_quantum_hparams(X_train, y_train, X_val, y_val)

    fmap = build_feature_map(best_cfg["map"], X_train.shape[1], best_cfg["reps"], best_cfg["ent"])
    model = train_quantum(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]), fmap, best_C)

    eval_and_log(model, X_test, y_test, "banknote", backend_name)


def run_two_moons(backend_name="qsvm_statevector"):
    df = generate_two_moons()
    df = df.sample(n=200, random_state=42) # Subsample for practical quantum run
    X_train, X_val, X_test, y_train, y_val, y_test = clean_split_scale(df)
    best_cfg, best_C = search_quantum_hparams(X_train, y_train, X_val, y_val)

    fmap = build_feature_map(best_cfg["map"], X_train.shape[1], best_cfg["reps"], best_cfg["ent"])
    model = train_quantum(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]), fmap, best_C)

    eval_and_log(model, X_test, y_test, "two_moons", backend_name)

if __name__ == "__main__":
    run_two_moons()
    run_banknote()