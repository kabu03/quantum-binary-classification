import sys, time, json, psutil, numpy as np
import os  # Make sure os is imported
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from qiskit_machine_learning.state_fidelities import ComputeUncompute

# primitives
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler as IBMSampler
from qiskit_aer.primitives import SamplerV2 as AerSampler
# Transpiler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import transpile

from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter

from src.data_utils import load_banknote, generate_two_moons, clean_split_scale
from src.metrics_logger import save_metrics, save_timing

# ---------------------------------------------------------------
# helper that makes every Parameter in a circuit unique & collision-free
def _freshen_parameters(circ, prefix="θ"):
    """Return a **copy** of `circ` whose Parameters are freshly renamed."""
    fresh = circ.copy()
    new_vec = ParameterVector(prefix, len(fresh.parameters))
    rename = {old: new for old, new in zip(fresh.parameters, new_vec)}
    return fresh.assign_parameters(rename, inplace=False)
# ---------------------------------------------------------------

# -------------- safety helper ---------------------------------
def check_memory(thresh_pct: int = 85):
    """Exit if RAM usage exceeds threshold percentage."""
    if psutil.virtual_memory().percent > thresh_pct:
        print(f"[MEM] RAM usage {psutil.virtual_memory().percent}% > {thresh_pct}% — aborting.")
        sys.exit(1)

def build_feature_map(kind: str, dim: int, reps: int, ent: str):
    """Builds Z or ZZ Feature Map manually using QuantumCircuit and ParameterVector."""
    # Use an explicit ParameterVector
    param_vec = ParameterVector('x', dim)  # Using 'x' prefix again
    print(f"[DEBUG] Using explicit ParameterVector: {param_vec.name}")

    qc = QuantumCircuit(dim, name=f"{kind}FeatureMapManual")

    for r in range(reps):
        # Initial Hadamard gates
        qc.h(range(dim))

        # Parameterized single-qubit rotations (Z or ZZ encoding)
        if kind == "Z":
            for i in range(dim):
                # ZFeatureMap uses U1(2*x_i) which is P(2*x_i) up to global phase.
                qc.p(2.0 * param_vec[i], i)
        elif kind == "ZZ":
            # First layer of Z rotations
            for i in range(dim):
                qc.p(2.0 * param_vec[i], i)

            # ZZ entanglement part based on ZZFeatureMap's default data_map_func
            # phi(x_i, x_j) = (pi - x_i) * (pi - x_j)
            # Apply RZZ(2 * phi(x_i, x_j)) which decomposes to CX, P, CX
            if ent == 'full':
                for i in range(dim):
                    for j in range(i + 1, dim):
                        qc.cx(i, j)
                        # Phase is 2.0 * (pi - x_i) * (pi - x_j)
                        phase = 2.0 * (np.pi - param_vec[i]) * (np.pi - param_vec[j])
                        qc.p(phase, j)  # Apply phase to target qubit
                        qc.cx(i, j)
            elif ent == 'linear':
                for i in range(dim - 1):
                    qc.cx(i, i + 1)
                    phase = 2.0 * (np.pi - param_vec[i]) * (np.pi - param_vec[i + 1])
                    qc.p(phase, i + 1)
                    qc.cx(i, i + 1)
            # Add other entanglement types if needed ('circular', etc.)
            else:
                raise ValueError(f"Unsupported ZZ entanglement type: {ent}")
        else:
            raise ValueError(f"Unknown feature map kind: {kind}")

        # Add barrier between reps unless it's the last rep
        if r < reps - 1:
            qc.barrier()

    print(f"[DEBUG] Manually built feature map circuit created with {qc.num_parameters} parameters.")
    # This manually constructed circuit still needs transpilation for the real backend
    return qc

def make_fidelity_kernel(fmap, backend_type: str):
    """
    Return FidelityQuantumKernel configured for the specified backend_type:
        backend_type='statevector' → noiseless local simulator (AerSampler)
        backend_type='qasm'        → noisy qasm simulator (AerSampler)
        backend_type=<real name>   → real IBM device via Runtime SamplerV2 (transpiled)
    """
    kernel_feature_map = fmap
    pass_manager = None
    backend = None # For real device case

    # --- Use the backend_type parameter ---
    if backend_type == "statevector":
        print(f"[INFO] Configuring AerSampler for statevector simulation.")
        sampler = AerSampler()
    elif backend_type == "qasm":
        print(f"[INFO] Configuring AerSampler for qasm simulation.")
        sampler = AerSampler()
    else:
        # ----- REAL DEVICE -----
        print(f"[INFO] Configuring SamplerV2 for real backend: {backend_type} using Session mode.")
        try:
            service = QiskitRuntimeService(channel="ibm_cloud") # Assumes token is loaded if needed
            backend = service.backend(backend_type)
            # ... (rest of real device setup: Session, IBMSampler, pass_manager, transpile) ...
            # Make sure to use backend_type in print/error messages
            pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
            transpiled_fmap = pass_manager.run(fmap)
            transpiled_fmap = _freshen_parameters(transpiled_fmap, prefix="θ")
            kernel_feature_map = transpiled_fmap
            # Define sampler for real device
            session = Session(backend=backend)
            options = {"default_shots": 100}
            sampler = IBMSampler(mode=session, options=options)

        except Exception as e:
            print(f"[ERROR] Failed to initialize/transpile for backend {backend_type}: {e}")
            raise RuntimeError(f"Could not configure/transpile for quantum backend {backend_type}.") from e
    # --- End backend configuration ---

    print(f"[INFO] Instantiating ComputeUncompute (Pass manager: {'Yes' if pass_manager else 'No'})")
    fidelity = ComputeUncompute(sampler=sampler, pass_manager=pass_manager)

    print("[INFO] Instantiating FidelityQuantumKernel.")
    return FidelityQuantumKernel(feature_map=kernel_feature_map, fidelity=fidelity)

# -------------- hyper-parameter search -------------------------
def search_quantum_hparams(X_train, y_train, X_val, y_val, backend_type:str):

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

            print("[DEBUG] Creating FidelityQuantumKernel...")
            quantum_kernel = make_fidelity_kernel(fmap, backend_type)
            print("[DEBUG] FidelityQuantumKernel created.")

            for C in [0.1, 1, 10]:
                print(f"[DEBUG] Loop for C={C}: Creating QSVC instance...")
                svc = QSVC(quantum_kernel=quantum_kernel, C=C)
                print(f"[DEBUG] Loop for C={C}: QSVC instance created.")

                print(f"[DEBUG] Loop for C={C}: Calling svc.fit...")
                svc.fit(X_train, y_train)
                print(f"[DEBUG] Loop for C={C}: svc.fit completed.")

                print(f"[DEBUG] Loop for C={C}: Calling svc.predict...")
                preds = svc.predict(X_val)
                print(f"[DEBUG] Loop for C={C}: svc.predict completed.")

                print("[DEBUG] Kernel matrices computed successfully (within fit/predict).")

                f1 = f1_score(y_val, preds)

                print(f"[TUNE] Config {cfg}, C={C} -> F1={f1:.3f}")
                if f1 > best_f1:
                    best_f1, best_cfg, best_C = f1, cfg, C

        except Exception as e:
            current_C = C if 'C' in locals() else 'N/A (before C loop or after error)'
            print(f"[ERROR] During config {cfg}, C={current_C}: {e}")

        finally:
            if 'svc' in locals():
                del svc
            import gc; gc.collect()
            print("[DEBUG] Cleaned up memory.")

    print(f"\n[INFO] Best configuration: {best_cfg} with C={best_C} and F1={best_f1:.3f}")
    return best_cfg, best_C

# -------------- training + evaluation --------------------------
# Add backend_type parameter here
def train_quantum(X_train, y_train, fmap, best_C, backend_type: str):
    """
    Train QSVC with the chosen feature-map, C, and backend_type.
    """
    print("\n[TRAIN] Training QSVC model...")
    # --- Pass backend_type to make_fidelity_kernel ---
    quantum_kernel = make_fidelity_kernel(fmap, backend_type)
    model = QSVC(quantum_kernel=quantum_kernel, C=best_C, verbose=True)

    t0 = time.time()
    model.fit(X_train, y_train)
    training_duration = time.time() - t0
    print("[TRAIN] Training completed.")

    return model, training_duration


def eval_and_log(model, training_duration, X_test, y_test, dataset_name, backend_type, run_dir):
    """
    Evaluate the trained model on test data, save results.
    Uses backend_type to construct the logging name.
    """
    # Construct the name used for logging/saving from the type
    log_backend_name = f"qsvm_{backend_type}"
    print(f"[EVAL] Evaluating backend type: {backend_type} (logging as: {log_backend_name}) on dataset: {dataset_name}")

    print("[EVAL] Making predictions...", flush=True)
    t0 = time.time()
    y_pred = model.predict(X_test)
    predicting_duration = time.time() - t0

    metrics = {
        "dataset": dataset_name,
        "backend": log_backend_name, # Use constructed name for consistency in logs
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion": confusion_matrix(y_test, y_pred).tolist()
    }
    print(json.dumps(metrics, indent=2))

    # Pass the constructed log_backend_name to savers
    save_metrics(metrics, log_backend_name, run_dir)
    save_timing(training_duration, predicting_duration, log_backend_name, dataset_name, run_dir)


# -------------- wrappers for each dataset ----------------------
def run_banknote(backend_type="statevector", num_samples=None, skip_hparam_search=True, run_dir = None):
    # Construct the display name early for print statements
    display_backend_name = f"qsvm_{backend_type}"
    print(f"\nProcessing dataset: banknote (quantum - {display_backend_name})") # Use display name
    df = load_banknote()
    if num_samples is not None and num_samples < len(df): # Use num_samples
        print(f"[INFO] Subsampling banknote dataset to {num_samples} samples.")
        df = df.sample(n=num_samples, random_state=42)

    X_train, X_val, X_test, y_train, y_val, y_test = clean_split_scale(df)

    if skip_hparam_search:
        print("[INFO] Skipping hyperparameter search, using default config.")
        best_cfg = {"map": "ZZ", "reps": 1, "ent": "full"}
        best_C = 1.0
        print(f"[INFO] Using fixed config: {best_cfg}, C={best_C}")
    else:
        print("[INFO] Starting hyperparameter search...")
        best_cfg, best_C = search_quantum_hparams(X_train, y_train, X_val, y_val, backend_type) # Added backend_type
        if best_cfg is None:
            raise RuntimeError(f"No quantum kernel could be built during hyperparameter search for backend '{backend_type}' – check configuration, dependencies, and network if applicable.")

    print(f"[INFO] Building feature map for chosen config: {best_cfg}")
    fmap = build_feature_map(best_cfg["map"], X_train.shape[1], best_cfg["reps"], best_cfg["ent"])

    X_final_train = np.vstack([X_train, X_val])
    y_final_train = np.hstack([y_train, y_val])

    model, training_duration = train_quantum(X_final_train, y_final_train, fmap, best_C, backend_type)

    eval_and_log(model, training_duration, X_test, y_test, "banknote", backend_type, run_dir)


def run_two_moons(backend_type="statevector", num_samples=None, skip_hparam_search=True, run_dir = None):
    # Construct the display name early for print statements
    display_backend_name = f"qsvm_{backend_type}"
    print(f"\nProcessing dataset: two_moons (quantum - {display_backend_name})") # Use display name
    df = generate_two_moons()
    if num_samples is not None and num_samples < len(df): # Use num_samples
        print(f"[INFO] Subsampling two_moons dataset to {num_samples} samples.")
        df = df.sample(n=num_samples, random_state=42)

    X_train, X_val, X_test, y_train, y_val, y_test = clean_split_scale(df)

    if skip_hparam_search:
        print("[INFO] Skipping hyperparameter search, using default config.")
        best_cfg = {"map": "ZZ", "reps": 1, "ent": "full"}
        best_C = 1.0
        print(f"[INFO] Using fixed config: {best_cfg}, C={best_C}")
    else:
        print("[INFO] Starting hyperparameter search...")
        best_cfg, best_C = search_quantum_hparams(X_train, y_train, X_val, y_val, backend_type) # Added backend_type
        if best_cfg is None:
            raise RuntimeError(f"No quantum kernel could be built during hyperparameter search for backend '{backend_type}' – check configuration, dependencies, and network if applicable.")

    print(f"[INFO] Building feature map for chosen config: {best_cfg}")
    fmap = build_feature_map(best_cfg["map"], X_train.shape[1], best_cfg["reps"], best_cfg["ent"])

    X_final_train = np.vstack([X_train, X_val])
    y_final_train = np.hstack([y_train, y_val])

    model, training_duration = train_quantum(X_final_train, y_final_train, fmap, best_C, backend_type)

    eval_and_log(model, training_duration, X_test, y_test, "two_moons", backend_type, run_dir)

if __name__ == "__main__":
    test_run_dir = os.path.join(os.path.dirname(__file__), "..", "results", "test_run_quantum")
    os.makedirs(test_run_dir, exist_ok=True)
    run_two_moons(backend_type="statevector", num_samples=100, skip_hparam_search=False, run_dir=test_run_dir)
    run_banknote(backend_type="qasm", num_samples=100, skip_hparam_search=False, run_dir=test_run_dir)