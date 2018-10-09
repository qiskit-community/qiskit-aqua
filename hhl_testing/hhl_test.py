from qiskit_aqua import run_algorithm
from hhl_test_suite import run_test
from qiskit_aqua.utils import random_hermitian
import numpy as np


matrix = random_hermitian(4, K=(5, 1, 1))
vector = np.random.random(4)

params = {
    "algorithm": {
        "name": "HHL",
        "mode": "state_tomography"
    },
    "eigs": {
        "name": "QPE",
        "num_time_slices": 50,
        "expansion_mode": "suzuki",
        "expansion_order": 2,
        "negative_evals": False,
        "num_ancillae": 5
    },
    "reciprocal": {
        "name": "LOOKUP",
        "lambda_min": 0.9,
    },
    "backend": {
        "name": "local_qasm_simulator",
        "shots": 1
    },
    "input": {
        "test_set": "test",
        "type": "generate",
        "n": 4,
        "cond": 10
    }
}


res = run_test(params)
print(res)
