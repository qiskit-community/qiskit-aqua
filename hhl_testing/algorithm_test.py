from qiskit_aqua import Operator, run_algorithm
from qiskit_aqua.input import get_input_instance
from qiskit_aqua._discover import local_algorithms
from qiskit_aqua import get_eigs_instance


params = {
    'algorithm': {'name': 'EigenvalueEstimation'},
    'eigs': {'name': 'QPE', 'num_ancillae': 6, 'negative_evals': True},
    'initial_state': {'name': 'CUSTOM', 'state_vector': [1/2**0.5, 1/2**0.5]},
    'backend': {'name': 'local_qasm_simulator'}
}

result = run_algorithm(params, [[-1, 1], [1, 2]])
print(result["visualization"]())

# from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
#
# import numpy as np
#
# qpe = get_eigs_instance("QPE")
#
# params = {
#     'num_ancillae': 2,
#     'num_time_slices': 1,
#     'expansion_mode': 'trotter',
#     'expansion_order': 1,
#     'hermitian_matrix': True,
#     'negative_evals': False,
#     'use_basis_gates': False,
#     'paulis_grouping': 'random',
#     'evo_time': np.pi/2,
#     "iqft": {'name': 'STANDARD'} 
# }
#
# qpe.init_params(params, [[3, 0], [0, 2]])
#
# q = QuantumRegister(2)
# qc = QuantumCircuit(q)
# qc.h(q)
#
# qc += qpe.construct_circuit('circuit', q)
#
# c = ClassicalRegister(len(qpe._output_register))
# qc.add(c)
# qc.measure(qpe._output_register, c)
#
# res = execute(qc, "local_qasm_simulator").result()
# print(res.get_counts())

# algo_input = get_input_instance('EnergyInput')
# algo_input.qubit_op = Operator(matrix=[[3, 0.01], [0.01, 1]])
#
# params = {
#     'algorithm': {'name': 'QPE', 'num_ancillae': 9},
#     'backend': {'name': 'local_qasm_simulator'}
# }
#
# result = run_algorithm(params, algo_input)
# print(result)
