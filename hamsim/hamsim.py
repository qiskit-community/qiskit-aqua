import sys
sys.path.append("..")

from qiskit import execute, QuantumRegister
from qiskit_aqua import Operator
from qiskit.tools.qi.qi import state_fidelity
from scipy.linalg import expm

import numpy as np

def hamsim(matrix, qr, evo_time, num_time_slices=1000, paulis_grouping="random", expansion_order=2):
    #setup operator
    op = Operator(matrix=matrix)
    pauli_list = op.reorder_paulis(grouping=paulis_grouping)
    #slice_pauli_list = pauli_list
    slice_pauli_list = op._suzuki_expansion_slice_pauli_list(pauli_list, 1, expansion_order)

    #handle global phase
    num_identities = 0
    for p in pauli_list:
        if np.all(p[1].v == 0) and np.all(p[1].w == 0):
            num_identities += 1
            if num_identities > 1:
                raise RuntimeError('Multiple identity pauli terms are present.')
            ancilla_phase_coef = p[0].real if isinstance(p[0], complex) else p[0]

    qc = op.construct_evolution_circuit(pauli_list, evo_time, num_time_slices, qr)

    #undo global phase
    lam = ancilla_phase_coef*evo_time
    qc.u1(lam, qr[0])
    qc.x(qr[0])
    qc.u1(lam, qr[0])
    qc.x(qr[0])
    return qc

def check_fidelity(matrix, evo_time, result, vec):
    theo = expm(1j*evo_time*matrix)
    return state_fidelity(theo.dot(vec), result.dot(vec))

matrix = np.array([[2, -1], [-1, 2]])
matrix = np.random.random([4, 4])
matrix = 1/2*(matrix+matrix.T)
print(matrix)
qr = QuantumRegister(2)

qc = hamsim(matrix, qr, np.pi/2)

res = execute(qc, "local_unitary_simulator")
res = res.result().get_data()["unitary"]

vec = np.random.random(4)
vec = vec/np.sqrt(vec.dot(vec.conj()))

print(check_fidelity(matrix, np.pi/2, res, vec))
