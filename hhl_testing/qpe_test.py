import numpy as np
from qiskit_aqua import run_algorithm

matrix = [[2, -1], [-1, 2]]
print(matrix)

w, v = np.linalg.eigh(matrix) 
print("Eigenvalues:", w)

invec = sum([v[:,i] for i in range(len(matrix))])
invec /= np.sqrt(invec.dot(invec.conj()))

params = {
    'algorithm': {
        'name': 'EigenvalueEstimation',
    },
    'eigs': {
        'name': 'QPE',
        'num_ancillae': 6,
        'num_time_slices': 1,
        'expansion_mode': 'trotter',
        'expansion_order': 1,
        'hermitian_matrix': True,
        'negative_evals': False,
    },
    "initial_state": {
        "name": "CUSTOM",
        "state_vector": list(invec)
    },
    "backend": {
        "name": "local_qasm_simulator",
    }
}

res = run_algorithm(params, matrix)
res["visualization"]()
