# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" controlled circuit """

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import Unroller


# pylint: disable=invalid-name
def apply_cp(circuit, lam, c, t, use_basis_gates=True):
    """ apply cp """
    if use_basis_gates:
        circuit.p(lam / 2, c)
        circuit.cx(c, t)
        circuit.p(-lam / 2, t)
        circuit.cx(c, t)
        circuit.p(lam / 2, t)
    else:
        circuit.cp(lam, c, t)


def apply_cu(circuit, theta, phi, lam, c, t, use_basis_gates=True):
    """ apply cu """
    if use_basis_gates:
        circuit.p((lam + phi) / 2, c)
        circuit.p((lam - phi) / 2, t)
        circuit.cx(c, t)
        circuit.u(-theta / 2, 0, -(phi + lam) / 2, t)
        circuit.cx(c, t)
        circuit.u(theta / 2, phi, 0, t)
    else:
        circuit.cu(theta, phi, lam, 0, c, t)


# pylint: disable=invalid-name
def apply_ccx(circuit, a, b, c, use_basis_gates=True):
    """ apply ccx """
    if use_basis_gates:
        circuit.h(c)
        circuit.cx(b, c)
        circuit.tdg(c)
        circuit.cx(a, c)
        circuit.t(c)
        circuit.cx(b, c)
        circuit.tdg(c)
        circuit.cx(a, c)
        circuit.t(b)
        circuit.t(c)
        circuit.h(c)
        circuit.cx(a, b)
        circuit.t(a)
        circuit.tdg(b)
        circuit.cx(a, b)
    else:
        circuit.ccx(a, b, c)


def get_controlled_circuit(circuit, ctl_qubit, tgt_circuit=None, use_basis_gates=True):
    """
    Construct the controlled version of a given circuit.

    Args:
        circuit (QuantumCircuit) : the base circuit
        ctl_qubit (Qubit) : the control qubit to use
        tgt_circuit (QuantumCircuit) : the target controlled circuit to be modified in-place
        use_basis_gates (bool) : boolean flag to indicate whether or not
                                only basis gates should be used

    Return:
        QuantumCircuit: a QuantumCircuit object with the base circuit being controlled by ctl_qubit
    Raises:
        RuntimeError: unexpected operation
    """
    if tgt_circuit is not None:
        qc = tgt_circuit
    else:
        qc = QuantumCircuit()

    # get all the qubits and clbits
    qregs = circuit.qregs
    qubits = []
    for qreg in qregs:
        if not qc.has_register(qreg):
            qc.add_register(qreg)
        qubits.extend(qreg)
    cregs = circuit.cregs
    clbits = []
    for creg in cregs:
        if not qc.has_register(creg):
            qc.add_register(creg)
        clbits.extend(creg)

    # get all operations
    unroller = Unroller(basis=['u', 'p', 'cx'])
    ops = dag_to_circuit(unroller.run(circuit_to_dag(circuit))).data

    # process all basis gates to add control
    if not qc.has_register(ctl_qubit._register):
        qc.add_register(ctl_qubit._register)
    for op in ops:
        if op[0].name == 'id':
            apply_cu(qc, 0, 0, 0, ctl_qubit, op[1][0], use_basis_gates=use_basis_gates)
        elif op[0].name == 'p':
            apply_cp(qc, *op[0].params, ctl_qubit, op[1][0], use_basis_gates=use_basis_gates)
        elif op[0].name == 'u':
            apply_cu(qc, *op[0].params, ctl_qubit, op[1][0], use_basis_gates=use_basis_gates)
        elif op[0].name == 'cx':
            apply_ccx(qc, ctl_qubit, op[1][0], op[1][1], use_basis_gates=use_basis_gates)
        elif op[0].name == 'measure':
            qc.measure(op[1], op[2])
        elif op[0].name == 'barrier':
            qc.barrier(op[1])
        else:
            raise RuntimeError('Unexpected operation {}.'.format(op[0].name))

    return qc
