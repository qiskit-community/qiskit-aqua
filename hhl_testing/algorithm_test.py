from qiskit import QuantumRegister, QuantumCircuit, execute
import qiskit.extensions.simulator

from qiskit_aqua import Operator, run_algorithm
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import get_eigs_instance, get_reciprocal_instance

from qiskit_aqua.utils import random_hermitian, random_non_hermitian

from qiskit.tools.visualization import plot_circuit

import numpy as np

# from qiskit_aqua.algorithms.components.reciprocals.lookup_rotation import LookupRotation
from qiskit_aqua.algorithms.single_sample.hhl.hhl import HHL

params = {
    "algorithm": {
        "name": "HHL",
        "mode": "debug"
    },
    "eigs": {
        "name": "QPE",
        "num_time_slices": 1,
        "expansion_mode": "trotter",
        "negative_evals": False,
        "num_ancillae": 5,
    },
    "initial_state": {
        "name": "CUSTOM",
        "state_vector": [1, 1]
    },
    "reciprocal": {
        "name": "GENCIRCUITS",
        "scale": 1
    },
    "backend": {
        "name": "local_qasm_simulator",
        "shots": 1
    }
}

# matrix = random_hermitian(2, eigrange=(1, 4), trunc=2)
matrix = np.array([[1, 0], [0, 3]])

ret = run_algorithm(params, matrix)
qc = ret["circuit"]
qc.snapshot("-1")

res = execute(qc, "local_qasm_simulator", config={"data":
    ["quantum_state_ket"]}, shots=100).result()
qsks = res.get_snapshot("1").get("quantum_state_ket")
print(qsks)
qsk = {}
for i in range(len(qsks)):
    for key in qsks[i]:
        if key[0] == "1":
            qsk[str(key[-1])] = qsks[i][str(key)][0] + 1j*qsks[i][str(key)][1]
try:
    v = np.array([qsk['0'], qsk['1']])
    print(v, matrix.dot(v))
except KeyError:
    raise KeyError("rotation seems to have failed")



