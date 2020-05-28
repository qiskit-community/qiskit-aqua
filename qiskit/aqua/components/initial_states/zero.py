# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The zero (null/vacuum) state."""

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua import AquaError
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.utils.validation import validate_min


class Zero(InitialState):
    """
    The zero (null/vacuum) state.

    This is suitable for those situations in which the all-zeros state is the
    desired state. This is the case for a *vacuum state* in physics or chemistry
    """

    def __init__(self, num_qubits: int) -> None:
        """
        Args:
            num_qubits: Number of qubits, has a minimum value of 1.
        """
        super().__init__()
        validate_min('num_qubits', num_qubits, 1)
        self._num_qubits = num_qubits

    def construct_circuit(self, mode='circuit', register=None):
        if mode == 'vector':
            return np.array([1.0] + [0.0] * (np.power(2, self._num_qubits) - 1))
        elif mode == 'circuit':
            if register is None:
                register = QuantumRegister(self._num_qubits, name='q')
            quantum_circuit = QuantumCircuit(register)
            return quantum_circuit
        else:
            raise AquaError('Mode should be either "vector" or "circuit"')
