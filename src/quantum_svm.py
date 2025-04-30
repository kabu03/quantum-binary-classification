import sys, time, json, psutil, numpy as np
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

# primitives
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler as IBMSampler # Ensure IBMSampler is SamplerV2
from qiskit_aer.primitives import Sampler as AerSampler
# quantum-kernel objects
from qiskit_machine_learning.kernels import FidelityQuantumKernel
# Transpiler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from src.data_utils import load_banknote, generate_two_moons, clean_split_scale
from src.metrics_logger import save_metrics, save_timing

import os
BACKEND = os.getenv("BACKEND", "statevector")

# -------------- safety helper ---------------------------------
def check_memory(thresh_pct: int = 85):
    """Exit if RAM usage exceeds threshold percentage."""
    if psutil.virtual_memory().percent > thresh_pct:
        print(f"[MEM] RAM usage {psutil.virtual_memory().percent}% > {thresh_pct}% — aborting.")
        sys.exit(1)

def build_feature_map(kind: str, dim: int, reps: int, ent: str):
    """Builds ZZFeatureMap or ZFeatureMap with a different prefix."""
    # Might use a different prefix to see if it avoids the specific 'x[1]' conflict
    prefix = 'x'
    print(f"[DEBUG] Using parameter prefix: {prefix}")
    if kind == "ZZ":
        fmap = ZZFeatureMap(feature_dimension=dim, reps=reps, entanglement=ent, parameter_prefix=prefix)
    elif kind == "Z":
        fmap = ZFeatureMap(feature_dimension=dim, reps=reps, parameter_prefix=prefix)
    else:
        raise ValueError(f"Unknown feature map kind: {kind}")

    # --- Ensure NO DECOMPOSITION ---
    return fmap # Return the original feature map

def make_fidelity_kernel(fmap):
    """
    Return FidelityQuantumKernel configured for the BACKEND env:
        BACKEND=statevector   → noiseless local simulator (AerSampler)
        BACKEND=qasm          → noisy qasm simulator (AerSampler)
        BACKEND=<real name>   → real IBM device via Runtime SamplerV2 (transpiled)
    """
    transpiled_fmap = fmap # Default for simulators

    if BACKEND == "statevector":
        sampler = AerSampler()
        kernel_feature_map = fmap
    elif BACKEND == "qasm":
        sampler = AerSampler()
        kernel_feature_map = fmap
    else:
        # ----- REAL DEVICE: Use SamplerV2 with Session mode AND TRANSPILE -----
        print(f"[INFO] Configuring SamplerV2 for real backend: {BACKEND} using Session mode.")
        try:
            service = QiskitRuntimeService(channel="ibm_cloud")
            backend = service.backend(BACKEND)
            session = Session(backend=backend)
            sampler = IBMSampler(mode=session)
            print(f"[INFO] SamplerV2 initialized successfully for backend {backend.name} via Session.")

            # <<< --- KEEP TRANSPILATION STEP --- >>>
            print(f"[INFO] Transpiling feature map for backend {backend.name} (Optimization Level 2)...")
            pm = generate_preset_pass_manager(backend=backend, optimization_level=2) # Use level 1
            # Transpile the ORIGINAL feature map
            transpiled_fmap = pm.run(fmap)
            print("[INFO] Feature map transpiled successfully.")
            # <<< --- END TRANSPILATION STEP --- >>>

            # Use the transpiled feature map for the kernel on real device
            kernel_feature_map = transpiled_fmap

        except Exception as e:
            print(f"[ERROR] Failed to initialize/transpile for backend {BACKEND}: {e}")
            raise RuntimeError(f"Could not configure/transpile for quantum backend {BACKEND}.") from e

    fidelity = ComputeUncompute(sampler=sampler)
    # Use the appropriate feature map (transpiled for real backend)
    return FidelityQuantumKernel(feature_map=kernel_feature_map, fidelity=fidelity)

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

            # <<< --- ADDED DEBUG --- >>>
            print("[DEBUG] Creating FidelityQuantumKernel...")
            quantum_kernel = make_fidelity_kernel(fmap)
            print("[DEBUG] FidelityQuantumKernel created.")
            # <<< --- END ADDED DEBUG --- >>>

            for C in [0.1, 1, 10]:
                # <<< --- ADDED DEBUG --- >>>
                print(f"[DEBUG] Loop for C={C}: Creating QSVC instance...")
                svc = QSVC(quantum_kernel=quantum_kernel, C=C)
                print(f"[DEBUG] Loop for C={C}: QSVC instance created.")
                # <<< --- END ADDED DEBUG --- >>>

                # <<< --- ADDED DEBUG --- >>>
                print(f"[DEBUG] Loop for C={C}: Calling svc.fit...")
                svc.fit(X_train, y_train)
                print(f"[DEBUG] Loop for C={C}: svc.fit completed.")
                # <<< --- END ADDED DEBUG --- >>>

                # <<< --- ADDED DEBUG --- >>>
                print(f"[DEBUG] Loop for C={C}: Calling svc.predict...")
                preds = svc.predict(X_val)
                print(f"[DEBUG] Loop for C={C}: svc.predict completed.")
                # <<< --- END ADDED DEBUG --- >>>

                print("[DEBUG] Kernel matrices computed successfully (within fit/predict).")

                f1 = f1_score(y_val, preds)

                print(f"[TUNE] Config {cfg}, C={C} -> F1={f1:.3f}")
                if f1 > best_f1:
                    best_f1, best_cfg, best_C = f1, cfg, C

        except Exception as e:
            # <<< --- MODIFIED DEBUG --- >>>
            current_C = C if 'C' in locals() else 'N/A (before C loop or after error)'
            print(f"[ERROR] During config {cfg}, C={current_C}: {e}")
            # <<< --- END MODIFIED DEBUG --- >>>

        finally:
            if 'svc' in locals():
                del svc
            import gc; gc.collect()
            print("[DEBUG] Cleaned up memory.")

    print(f"\n[INFO] Best configuration: {best_cfg} with C={best_C} and F1={best_f1:.3f}")
    return best_cfg, best_C

# -------------- training + evaluation --------------------------
def train_quantum(X_train, y_train, fmap, best_C):
    """
    Train QSVC with the chosen feature-map and C.
    """
    print("\n[TRAIN] Training QSVC model...")
    quantum_kernel = make_fidelity_kernel(fmap)
    model = QSVC(quantum_kernel=quantum_kernel, C=best_C, verbose=True)

    t0 = time.time()
    model.fit(X_train, y_train)
    training_duration = time.time() - t0
    print("[TRAIN] Training completed.")

    return model, training_duration


def eval_and_log(model, training_duration, X_test, y_test, dataset_name, backend_name):
    """
    Evaluate the trained model on test data, save results.
    """

    print("[EVAL] Making predictions...", flush=True)
    t0 = time.time()
    y_pred = model.predict(X_test)
    predicting_duration = time.time() - t0

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
    save_timing(training_duration, predicting_duration, backend_name, dataset_name)


# -------------- wrappers for each dataset ----------------------
def run_banknote(backend_name="qsvm_statevector", skip_hparam_search=True): # Added flag
    print(f"\nProcessing dataset: banknote (skip_hparam_search={skip_hparam_search})")
    df = load_banknote()
    df = df.sample(n=50, random_state=42)  # Subsample for test quantum run

    X_train, X_val, X_test, y_train, y_val, y_test = clean_split_scale(df)

    if skip_hparam_search:
        print("[INFO] Skipping hyperparameter search, using default config.")
        # --- Use a fixed default configuration ---
        best_cfg = {"map": "ZZ", "reps": 1, "ent": "full"} # Example default
        best_C = 1.0 # Example default C
        print(f"[INFO] Using fixed config: {best_cfg}, C={best_C}")
        # -----------------------------------------
    else:
        print("[INFO] Starting hyperparameter search...")
        best_cfg, best_C = search_quantum_hparams(X_train, y_train, X_val, y_val)
        if best_cfg is None:
            raise RuntimeError("No quantum kernel could be built during hyperparameter search – check BACKEND and Internet.")

    # --- Proceed with training and evaluation ---
    print(f"[INFO] Building feature map for chosen config: {best_cfg}")
    fmap = build_feature_map(best_cfg["map"], X_train.shape[1], best_cfg["reps"], best_cfg["ent"])

    # Combine train and validation sets for final training
    X_final_train = np.vstack([X_train, X_val])
    y_final_train = np.hstack([y_train, y_val])

    model, training_duration = train_quantum(X_final_train, y_final_train, fmap, best_C)

    eval_and_log(model, training_duration, X_test, y_test, "banknote", backend_name)


def run_two_moons(backend_name="qsvm_statevector", skip_hparam_search=True): # Added flag
    print(f"\nProcessing dataset: two_moons (skip_hparam_search={skip_hparam_search})")
    df = generate_two_moons()
    df = df.sample(n=50, random_state=42)  # Subsample for test quantum run
    X_train, X_val, X_test, y_train, y_val, y_test = clean_split_scale(df)

    if skip_hparam_search:
        print("[INFO] Skipping hyperparameter search, using default config.")
        # --- Use a fixed default configuration ---
        best_cfg = {"map": "ZZ", "reps": 1, "ent": "full"} # Example default
        best_C = 1.0 # Example default C
        print(f"[INFO] Using fixed config: {best_cfg}, C={best_C}")
        # -----------------------------------------
    else:
        print("[INFO] Starting hyperparameter search...")
        best_cfg, best_C = search_quantum_hparams(X_train, y_train, X_val, y_val)
        if best_cfg is None:
            raise RuntimeError("No quantum kernel could be built during hyperparameter search – check BACKEND and Internet.")

    # --- Proceed with training and evaluation ---
    print(f"[INFO] Building feature map for chosen config: {best_cfg}")
    fmap = build_feature_map(best_cfg["map"], X_train.shape[1], best_cfg["reps"], best_cfg["ent"])

    # Combine train and validation sets for final training
    X_final_train = np.vstack([X_train, X_val])
    y_final_train = np.hstack([y_train, y_val])

    model, training_duration = train_quantum(X_final_train, y_final_train, fmap, best_C)

    eval_and_log(model, training_duration, X_test, y_test, "two_moons", backend_name)

if __name__ == "__main__":
    # Example: Run with skipping hyperparameter search
    run_two_moons(skip_hparam_search=True)
    run_banknote(skip_hparam_search=True)

    # Example: Run with hyperparameter search (if you want to test it)
    # run_two_moons(skip_hparam_search=False)
    # run_banknote(skip_hparam_search=False)