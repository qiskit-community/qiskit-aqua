# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Controlled-RY (cry) and Multiple-Control RY (mcry) Gates
"""

import logging

from qiskit import QuantumCircuit, QuantumRegister

from qiskit.aqua import AquaError
from qiskit.aqua.utils.circuit_utils import is_qubit

logger = logging.getLogger(__name__)


def cry(self, theta, q_control, q_target):
    """
    Apply Controlled-RY (cry) Gate.

    Args:
        self (QuantumCircuit): The circuit to apply the ch gate on.
        theta (float): The rotation angle.
        q_control ((QuantumRegister, int)): The control qubit.
        q_target ((QuantumRegister, int)): The target qubit.
    """

    if not is_qubit(q_control):
        raise AquaError('A qubit is expected for the control.')
    if not self.has_register(q_control[0]):
        raise AquaError('The control qubit is expected to be part of the circuit.')

    if not is_qubit(q_target):
        raise AquaError('A qubit is expected for the target.')
    if not self.has_register(q_target[0]):
        raise AquaError('The target qubit is expected to be part of the circuit.')

    if q_control == q_target:
        raise AquaError('The control and target need to be different qubits.')

    self.u3(theta / 2, 0, 0, q_target)
    self.cx(q_control, q_target)
    self.u3(-theta / 2, 0, 0, q_target)
    self.cx(q_control, q_target)
    return self


def mcry(self, theta, q_controls, q_target, q_ancillae):
    """
    Apply Multiple-Control RY (mcry) Gate.

    Args:
        self (QuantumCircuit): The circuit to apply the ch gate on.
        theta (float): The rotation angle.
        q_controls (QuantumRegister | (QuantumRegister, int)): The control qubits.
        q_target ((QuantumRegister, int)): The target qubit.
        q_ancillae (QuantumRegister | (QuantumRegister, int)): The ancillary qubits.
    """

    # check controls
    if isinstance(q_controls, QuantumRegister):
        control_qubits = [qb for qb in q_controls]
    elif isinstance(q_controls, list):
        control_qubits = q_controls
    else:
        raise AquaError('The mcry gate needs a list of qubits or a quantum register for controls.')

    # check target
    if is_qubit(q_target):
        target_qubit = q_target
    else:
        raise AquaError('The mcry gate needs a single qubit as target.')

    # check ancilla
    if q_ancillae is None:
        ancillary_qubits = []
    elif isinstance(q_ancillae, QuantumRegister):
        ancillary_qubits = [qb for qb in q_ancillae]
    elif isinstance(q_ancillae, list):
        ancillary_qubits = q_ancillae
    else:
        raise AquaError('The mcry gate needs None or a list of qubits or a quantum register for ancilla.')

    all_qubits = control_qubits + [target_qubit] + ancillary_qubits

    self._check_qargs(all_qubits)
    self._check_dups(all_qubits)

    self.u3(theta / 2, 0, 0, q_target)
    self.mct(q_controls, q_target, q_ancillae)
    self.u3(-theta / 2, 0, 0, q_target)
    self.mct(q_controls, q_target, q_ancillae)
    return self


QuantumCircuit.cry = cry
QuantumCircuit.mcry = mcry
