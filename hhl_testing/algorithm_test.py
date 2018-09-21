from qiskit_aqua import Operator, run_algorithm
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import get_eigs_instance

from qiskit_aqua.utils import random_hermitian, random_non_hermitian

params = {
    'algorithm': {'name': 'EigenvalueEstimation'},
    'eigs': {
        'name': 'QPE', 
        'num_ancillae': 6, 
        'negative_evals': False,
        'hermitian_matrix': True,
        'num_time_slices': 100,
        'expansion_mode': 'trotter',
        'expansion_order': '1'
    },
    'initial_state': {'name': 'CUSTOM', 'state_vector': [1/2**0.5, 1/2**0.5]},
    'backend': {'name': 'local_qasm_simulator'}
}

result = run_algorithm(params, random_hermitian(2, eigrange=(0, 5)))
result["visualization"]()
