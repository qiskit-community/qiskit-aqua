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
Controlled-Hadamard (ch) Gate.
"""

import logging
from math import pi

from qiskit import QuantumCircuit

from qiskit.aqua import AquaError
from qiskit.aqua.utils.circuit_utils import is_qubit

logger = logging.getLogger(__name__)


def ch(self, q_control, q_target):
    """
    Apply Controlled-Hadamard (ch) Gate.

    Note that this implementation of the ch uses a single cx gate,
    which is more efficient than what's currently provided in Terra.

    Args:
        self (QuantumCircuit): The circuit to apply the ch gate on.
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

    self.u3(-7 / 4 * pi, 0, 0, q_target)
    self.cx(q_control, q_target)
    self.u3(7 / 4 * pi, 0, 0, q_target)
    return self


QuantumCircuit.ch = ch
