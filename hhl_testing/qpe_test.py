from qiskit_aqua import run_algorithm
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import get_eigs_instance

from qiskit_aqua.utils import random_hermitian, random_non_hermitian

import numpy as np

matrix = random_hermitian(2, eigrange=(-1, 5))

if np.allclose(matrix, matrix.T.conj()):
    w, v = np.linalg.eigh(matrix)
    negative_evals = min(w) < 0
    hermitian_matrix = True
    invec = sum([v[:,i] for i in range(len(w))])
else:
    u, s, v = np.linalg.svd(matrix)
    negative_evals = True
    hermitian_matrix = False
    invec = matrix.shape[0]*[1]

invec /= np.sqrt(invec.dot(invec.conj()))
print(invec.dtype)
if True:
    invec = list(map(lambda x: [x.real, x.imag], invec))

invec = [[0, 1], [0, 0]]

params = {
    'algorithm': {'name': 'EigenvalueEstimation'},
    'eigs': {
        'name': 'QPE', 
        'num_ancillae': 6, 
        'negative_evals': negative_evals,
        'hermitian_matrix': hermitian_matrix,
        'num_time_slices': 100,
        'expansion_mode': 'trotter',
        'expansion_order': '1'
    },
    'initial_state': {'name': 'CUSTOM', 'state_vector': invec},
    'backend': {'name': 'local_qasm_simulator'}
}

result = run_algorithm(params, matrix)
result["visualization"]()
