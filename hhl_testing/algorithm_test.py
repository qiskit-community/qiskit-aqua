from qiskit_aqua import Operator, run_algorithm
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import get_eigs_instance, get_reciprocal_instance

from qiskit_aqua.algorithms.single_sample.hhl.hhl import HHL

from qiskit_aqua.utils import random_hermitian, random_non_hermitian

from qiskit.tools.visualization import plot_circuit

params = {
    "algorithm": {
        "name": "HHL"
    },
    "eigs": {
        "name": "QPE",
        "num_time_slices": 1,
        "expansion_mode": "trotter",
        "negative_evals": False,
        "num_ancillae": 3,
    }
}

matrix = random_hermitian(2, trunc=2)

qc = run_algorithm(params, matrix)
