# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
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

from qiskit import QuantumRegister, QuantumCircuit
import numpy as np

from qiskit_aqua.utils.initial_states import InitialState


class Zero(InitialState):
    """A zero (null/vacuum) state."""

    ZERO_CONFIGURATION = {
        'name': 'ZERO',
        'description': 'Zero initial state',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'zero_state_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.ZERO_CONFIGURATION.copy())
        self._num_qubits = 0

    def init_args(self, num_qubits):
        self._num_qubits = num_qubits

    def construct_circuit(self, mode, register=None):
        if mode == 'vector':
            return np.array([1.0] + [0.0] * (np.power(2, self._num_qubits) - 1))
        elif mode == 'circuit':
            if register is None:
                register = QuantumRegister(self._num_qubits, name='q')
            quantum_circuit = QuantumCircuit(register)
            return quantum_circuit
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')
