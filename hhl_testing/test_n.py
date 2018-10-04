from qiskit_aqua import run_algorithm
from hhl_test import get_matrix, get_vector
import matplotlib.pyplot as plt

params = {
    "algorithm": {
        "name": "HHL",
        "mode": "debug"
    },
    "eigs": {
        "name": "QPE",
        "num_time_slices": 100,
        "expansion_mode": "trotter",
        "expansion_order": 1,
        "negative_evals": False,
        "num_ancillae": 4,
    },
    "reciprocal": {
        "name": "LOOKUP",
        "lambda_min": 1,
    },
    "backend": {
        "name": "local_qasm_simulator",
        "shots": 1
    }
}

res = run_algorithm(params, (get_matrix(1), get_vector(3)))
