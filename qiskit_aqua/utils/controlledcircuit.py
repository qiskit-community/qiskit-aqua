# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np

from qiskit import QuantumCircuit
from qiskit.wrapper import get_backend
from qiskit import transpiler


def apply_cu1(circuit, lam, c, t, use_basis_gates=True):
    if use_basis_gates:
        circuit.u1(lam / 2, c)
        circuit.cx(c, t)
        circuit.u1(-lam / 2, t)
        circuit.cx(c, t)
        circuit.u1(lam / 2, t)
    else:
        circuit.cu1(lam, c, t)


def apply_cu3(circuit, theta, phi, lam, c, t, use_basis_gates=True):
    if use_basis_gates:
        circuit.u1((lam - phi) / 2, t)
        circuit.cx(c, t)
        circuit.u3(-theta / 2, 0, -(phi + lam) / 2, t)
        circuit.cx(c, t)
        circuit.u3(theta / 2, phi, 0, t)
    else:
        circuit.cu3(theta, phi, lam, c, t)
    # the u3 gate below is added to account for qiskit terra's cu3
    circuit.u3(0, 0, (phi + lam) / 2, c)


def apply_ccx(circuit, a, b, c, use_basis_gates=True):
    if use_basis_gates:
        circuit.u2(0, np.pi, c)
        circuit.cx(b, c)
        circuit.u1(-np.pi / 4, c)
        circuit.cx(a, c)
        circuit.u1(np.pi / 4, c)
        circuit.cx(b, c)
        circuit.u1(-np.pi / 4, c)
        circuit.cx(a, c)
        circuit.u1(np.pi / 4, b)
        circuit.u1(np.pi / 4, c)
        circuit.u2(0, np.pi, c)
        circuit.cx(a, b)
        circuit.u1(np.pi / 4, a)
        circuit.u1(-np.pi / 4, b)
        circuit.cx(a, b)
    else:
        circuit.ccx(a, b, c)


def get_controlled_circuit(circuit, ctl_qubit, tgt_circuit=None, use_basis_gates=True):
    """
    Construct the controlled version of a given circuit.

    Args:
        circuit (QuantumCircuit) : the base circuit
        ctl_qubit (indexed QuantumRegister) : the control qubit to use
        tgt_circuit (QuantumCircuit) : the target controlled circuit to be modified in-place
        use_basis_gates (bool) : boolean flag to indicate whether or not only basis gates should be used

    Return:
        a QuantumCircuit object with the base circuit being controlled by ctl_qubit
    """
    if tgt_circuit is not None:
        qc = tgt_circuit
    else:
        qc = QuantumCircuit()

    # get all the qubits and clbits
    qregs = circuit.get_qregs()
    qubits = []
    for name in qregs:
        if not qc.has_register(qregs[name]):
            qc.add(qregs[name])
        qubits.extend(qregs[name])
    cregs = circuit.get_cregs()
    clbits = []
    for name in cregs:
        if not qc.has_register(cregs[name]):
            qc.add(cregs[name])
        clbits.extend(cregs[name])

    # get all operations from compiled circuit
    ops = transpiler.compile(
        circuit,
        get_backend('local_qasm_simulator'),
        basis_gates='u1,u2,u3,cx,id'
    )['circuits'][0]['compiled_circuit']['operations']

    # process all basis gates to add control
    if not qc.has_register(ctl_qubit[0]):
        qc.add(ctl_qubit[0])
    for op in ops:
        if op['name'] == 'id':
            apply_cu3(qc, 0, 0, 0, ctl_qubit, qubits[op['qubits'][0]], use_basis_gates=use_basis_gates)
        elif op['name'] == 'u1':
            apply_cu1(qc, *op['params'], ctl_qubit, qubits[op['qubits'][0]], use_basis_gates=use_basis_gates)
        elif op['name'] == 'u2':
            apply_cu3(qc, np.pi / 2, *op['params'], ctl_qubit, qubits[op['qubits'][0]], use_basis_gates=use_basis_gates)
        elif op['name'] == 'u3':
            apply_cu3(qc, *op['params'], ctl_qubit, qubits[op['qubits'][0]], use_basis_gates=use_basis_gates)
        elif op['name'] == 'cx':
            apply_ccx(qc, ctl_qubit, *[qubits[i] for i in op['qubits']], use_basis_gates=use_basis_gates)
        elif op['name'] == 'measure':
            qc.measure(qubits[op['qubits'][0]], clbits[op['clbits'][0]])
        else:
            raise RuntimeError('Unexpected operation {}.'.format(op['name']))

    return qc
