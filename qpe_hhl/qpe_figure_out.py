#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:41:41 2018

@author: gawel
"""
import logging

from functools import reduce
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.tools.qi.pauli import Pauli
from qiskit_aqua import Operator, QuantumAlgorithm, AlgorithmError
from qiskit_aqua import get_initial_state_instance, get_iqft_instance
from qiskit.tools.visualization import circuit_drawer
from qiskit.tools.visualization._circuit_visualization import matplotlib_circuit_drawer

logger = logging.getLogger(__name__)

#from qpe import QPE
from qiskit_aqua import Operator

import numpy as np

import matplotlib.pyplot as plt

#qpe = QPE()

w = [-1, -1]
n = 2           #matrix dimensions

while min(w) <= 0:
    matrix = np.random.random([n, n])+1j*np.random.random([n, n])
    matrix = 4*(matrix+matrix.T.conj())
    op = Operator(matrix=matrix)
    w, v = np.linalg.eig(matrix)
    
matrix = np.diag([2,4])
operator = Operator(matrix=matrix)
w, v = np.linalg.eig(matrix)

print('matrix: \n', matrix)
print('condition number \n', max(abs(w))/min(abs(w)))
print('eigenvalues (Re):', w.real)
print('eigenvectors: \n', v)
#%%
invec = sum([v[:,i] for i in range(n)])
invec /= np.sqrt(invec.dot(invec.conj()))

params = {
'algorithm': {
        'name': 'QPE',
        #'num_ancillae':4 ,
        'num_time_slices': 10,
        'expansion_mode': 'suzuki',
        'expansion_order': 3,
        #'evo_time': np.pi/2
},
"iqft": {
    "name": "STANDARD"
},
"initial_state": {
    "name": "CUSTOM",
    "state_vector": invec#[1/2**0.5,1/2**0.5]
}}

num_time_slices = 10
num_ancillae = 4
paulis_grouping = 'random'
expansion_mode = 'suzuki'
expansion_order = 3
evo_time = None ##
ancilla_phase_coef = 0

ret = {}


#state_in = invec
init_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)
init_state_params['num_qubits'] = operator.num_qubits
init_state = get_initial_state_instance(init_state_params['name'])
init_state.init_params(init_state_params)
#print(init_state_params)
state_in = init_state

iqft_params = params.get(QuantumAlgorithm.SECTION_KEY_IQFT)
iqft_params['num_qubits'] = num_ancillae
iqft = get_iqft_instance(iqft_params['name'])
iqft.init_params(iqft_params)
#%%
# =============================================================================
# qpe.init_params(params, op) 
# =============================================================================

operator._check_representation('paulis')
if evo_time == None:
    evo_time = (1-2**-num_ancillae)*2*np.pi/sum([abs(p[0]) for p in operator.paulis])


# check for identify paulis to get its coef for applying global phase shift on ancillae later
num_identities = 0
for p in operator.paulis:
    if np.all(p[1].v == 0) and np.all(p[1].w == 0):
        num_identities += 1
        if num_identities > 1:
            raise RuntimeError('Multiple identity pauli terms are present.')
        ancilla_phase_coef = p[0].real if isinstance(p[0], complex) else p[0]




#%%
# =============================================================================
#     def _construct_phase_estimation_circuit(self, measure=False):
#         """Implement the Quantum Phase Estimation algorithm"""
# =============================================================================
a = QuantumRegister(num_ancillae, name='a')
q = QuantumRegister(operator.num_qubits, name='q')
qc = QuantumCircuit(a, q)
measure = False
if measure:
    c = ClassicalRegister(num_ancillae, name='c')
    qc.add(c)
    
qc += state_in.construct_circuit('circuit', q)
qc.barrier(q)

# Put all ancillae in uniform superposition
qc.u2(0, np.pi, a)
#circuit_drawer(qc)
#%%


# phase kickbacks via dynamics
pauli_list = operator.reorder_paulis(grouping=paulis_grouping)
if len(pauli_list) == 1:
    slice_pauli_list = pauli_list
else:
    if expansion_mode == 'trotter':
        slice_pauli_list = pauli_list
    elif expansion_mode == 'suzuki':
        slice_pauli_list = Operator._suzuki_expansion_slice_pauli_list(
            pauli_list,
            1,
            expansion_order
        )
    else:
        raise ValueError('Unrecognized expansion mode {}.'.format(expansion_mode))

#%%
for i in range(num_ancillae):
    qc += operator.construct_evolution_circuit(
        slice_pauli_list, evo_time, num_time_slices, q, a,
        ctl_idx=i, use_basis_gates=True )
    # global phase shift for the ancilla due to the identity pauli term
    if ancilla_phase_coef > 0:
        qc.u1(evo_time * ancilla_phase_coef * (2 ** i), a[i])
        
#%%        
iqft.construct_circuit('circuit', a, qc)

circuit = qc 

result = execute(circuit, backend="local_qasm_simulator", shots = 100).result()
print(result)
counts = result.get_counts(circuit)

rd = result.get_counts(circuit)
rets = sorted([[rd[k], k, k] for k in rd])[::-1]
for d in rets:
    d[2] = sum([2**-(i+1) for i, e in enumerate(reversed(d[2])) if e == "1"])*2*np.pi/evo_time

ret['measurements'] = rets
ret['evo_time'] = evo_time

#%%
print(ret["measurements"])
x = []
y = []
for c, k, l in ret["measurements"]:
    x += [l]
    y += [c]

plt.bar(x, y, width=0.1)
plt.show()