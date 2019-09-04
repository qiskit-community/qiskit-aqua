# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A zero (null/vacuum) state."""

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua import AquaError
from qiskit.aqua.components.initial_states import InitialState


class Zero(InitialState):
    """A zero (null/vacuum) state."""

    CONFIGURATION = {
        'name': 'ZERO',
        'description': 'Zero initial state',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'zero_state_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits):
        """Constructor.

        Args:
            num_qubits (int): number of qubits.
        """
        super().__init__()
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
