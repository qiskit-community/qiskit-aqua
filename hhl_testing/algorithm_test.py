from qiskit import QuantumRegister, QuantumCircuit, execute
import qiskit.extensions.simulator

from qiskit_aqua import Operator, run_algorithm
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import get_eigs_instance, get_reciprocal_instance

from qiskit_aqua.utils import random_hermitian, random_non_hermitian

from qiskit.tools.visualization import plot_circuit

import numpy as np

# from qiskit_aqua.algorithms.components.reciprocals.lookup_rotation import LookupRotation

params = {
    "algorithm": {
        "name": "HHL"
    },
    "eigs": {
        "name": "QPE",
        "num_time_slices": 1,
        "expansion_mode": "trotter",
        "negative_evals": False,
        "num_ancillae": 8,
    },
    "initial_state": {
        "name": "CUSTOM",
        "state_vector": [1, 1]
    },
    "reciprocal": {
        "name": "LOOKUP",
        "scale": 2**6
    }
}

matrix = random_hermitian(2, eigrange=(1, 4), trunc=2)
print(np.linalg.eig(matrix))
# matrix = np.array([[1, 0], [0, 3]])

qc = run_algorithm(params, matrix)
# plot_circuit(qc)
qc.snapshot("-1")

res = execute(qc, "local_qasm_simulator", config={"data":
    ["quantum_state_ket"]}, shots=100).result()
# print(res.get_snapshot("1").get("quantum_state_ket"))
# print()
qsks = res.get_snapshot("-1").get("quantum_state_ket")
qsk = next(e for e in qsks if list(e.keys())[0][0] == "1")
qsk = {k[-1]: v[0]+1j*v[1] for k, v in qsk.items()}
v = np.array([qsk['0'], qsk['1']])
print(matrix)
print(np.linalg.inv(matrix).dot(np.array([1, 1])))
print(v, matrix.dot(v))


