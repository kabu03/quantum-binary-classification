import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

# --- Configuration ---
BACKEND_NAME = "ibm_sherbrooke" # Or "ibm_brisbane"
NUM_QUBITS = 4 # Match your data dimension
PARAM_PREFIX = 'x'
# ---------------------

print(f"Testing transpilation + assign_parameters for backend: {BACKEND_NAME}")

# 1. Create a simple parameterized circuit (similar structure to ZZFeatureMap)
print(f"\n1. Creating ParameterVector '{PARAM_PREFIX}' with dim={NUM_QUBITS}...")
params = ParameterVector(PARAM_PREFIX, NUM_QUBITS)
qc = QuantumCircuit(NUM_QUBITS, name="MinimalParamCircuit")

print("   Applying H gates...")
qc.h(range(NUM_QUBITS))
print("   Applying P gates...")
for i in range(NUM_QUBITS):
    qc.p(2.0 * params[i], i)
print("   Applying CX and P gates (linear entanglement)...")
for i in range(NUM_QUBITS - 1):
    qc.cx(i, i + 1)
    phase = 2.0 * (np.pi - params[i]) * (np.pi - params[i+1])
    qc.p(phase, i + 1)
    qc.cx(i, i + 1)

print(f"   Original circuit created with {qc.num_parameters} parameters: {qc.parameters}")
# print(qc.draw(output='text')) # Optional: print circuit

# 2. Get backend and transpile
print(f"\n2. Transpiling for {BACKEND_NAME} (Optimization Level 1)...")
try:
    # Use channel="ibm_cloud" as "ibm_quantum" is deprecated
    service = QiskitRuntimeService(channel="ibm_cloud")
    backend = service.backend(BACKEND_NAME)
    print(f"   Got backend: {backend.name}")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled_qc = pm.run(qc)
    print(f"   Transpilation successful.")
    print(f"   Transpiled circuit has {transpiled_qc.num_parameters} parameters: {transpiled_qc.parameters}")
    # print(transpiled_qc.draw(output='text')) # Optional: print transpiled circuit

except Exception as e:
    print(f"[ERROR] Failed during backend retrieval or transpilation: {e}")
    exit()

# 3. Try to assign parameters to the *transpiled* circuit
print(f"\n3. Attempting to assign parameters to the *transpiled* circuit...")
dummy_values = np.random.rand(NUM_QUBITS)
# Create the dictionary mapping Parameter objects to values
param_dict = {param: val for param, val in zip(params, dummy_values)}
# Alternative: Create dict using string names (less robust but sometimes needed)
# param_dict_str = {str(param): val for param, val in zip(params, dummy_values)}


try:
    # This is the step that fails in the traceback
    print(f"   Assigning using Parameter objects: {list(param_dict.keys())}")
    bound_circuit = transpiled_qc.assign_parameters(param_dict)
    # If successful, this line will be reached
    print(f"   SUCCESS: assign_parameters completed.")
    print(f"   Bound circuit has {bound_circuit.num_parameters} parameters.")

    # Optional: Try assigning with string keys if the above worked but fails elsewhere
    # print(f"\n   Assigning using string keys: {list(param_dict_str.keys())}")
    # bound_circuit_str = transpiled_qc.assign_parameters(param_dict_str)
    # print(f"   SUCCESS: assign_parameters with string keys completed.")

except Exception as e:
    print(f"[ERROR] Failed during assign_parameters on transpiled circuit: {e}")
    print("   This confirms the incompatibility.")

print("\nTest finished.")