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

from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit  # pylint: disable=unused-import

from qiskit.aqua import AquaError

logger = logging.getLogger(__name__)


def cry(self, theta, q_control, q_target):
    """
    Apply Controlled-RY (cry) Gate.

    Args:
        self (QuantumCircuit): The circuit to apply the cry gate on.
        theta (float): The rotation angle.
        q_control (Union(Qubit, int)): The control qubit.
        q_target (Union(Qubit, int)): The target qubit.
    Returns:
        QuantumCircuit: instance self
    Raises:
        AquaError: invalid input
    """

    qubits = [q_control, q_target]
    names = ["control", "target"]

    for qubit, name in zip(qubits, names):
        if isinstance(qubit, Qubit):
            if not self.has_register(q_control.register):
                raise AquaError('The {} qubit is expected to be part of the circuit.'.format(name))
        elif isinstance(qubit, int):
            if qubit >= self.n_qubits:
                raise AquaError('Qubit index out of range.')
        else:
            raise AquaError('A qubit or int is expected for the {}.'.format(name))

    if q_control == q_target:
        raise AquaError('The control and target need to be different qubits.')

    self.u3(theta / 2, 0, 0, q_target)
    self.cx(q_control, q_target)
    self.u3(-theta / 2, 0, 0, q_target)
    self.cx(q_control, q_target)
    return self


QuantumCircuit.cry = cry
