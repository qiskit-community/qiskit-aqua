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
        "mode": "state_tomography"
    },
    "eigs": {
        "name": "QPE",
        "num_time_slices": 1,
        "expansion_mode": "trotter",
        "negative_evals": False,
        "num_ancillae": 8,
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

matrix = np.array([[1, 0], [0, 2]])
# matrix = random_hermitian(2, eigrange=(1, 4))

ret = run_algorithm(params, (matrix, [1, 1]))
print(ret)
# print(matrix.dot(ret["solution"]))

