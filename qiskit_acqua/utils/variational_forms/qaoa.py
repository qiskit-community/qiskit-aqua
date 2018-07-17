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

import numpy as np
from functools import reduce

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.tools.qi.pauli import Pauli

from qiskit_acqua.operator import Operator
from qiskit_acqua.utils.variational_forms import VariationalForm


class VarFormQAOA(VariationalForm):
    """Global X phases and parameterized problem hamiltonian."""

    QAOA_VF_CONFIGURATION = {
        'name': 'QAOA',
        'description': 'QAOA Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'qaoa_vf_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.QAOA_VF_CONFIGURATION.copy())
        self._depth = 0
        self._initial_state = None

    def init_args(self, cost_operator, depth, initial_state=None):
        self._cost_operator = cost_operator
        self._depth = depth
        self._num_parameters = 2 * depth
        self._bounds = [(0, np.pi)] * depth + [(0, 2 * np.pi)] * depth
        self._preferred_init_points = [0] * depth * 2
        self._initial_state = initial_state

        # prepare the mixer operator
        v = np.zeros(self._cost_operator.num_qubits)
        ws = np.eye(self._cost_operator.num_qubits)
        self._mixer_operator = reduce(
            lambda x, y: x + y,
            [
                Operator([[1, Pauli(v, ws[i, :])]])
                for i in range(self._cost_operator.num_qubits)
            ]
        )

    def construct_circuit(self, angles):
        if not len(angles) == self.num_parameters:
            raise ValueError('Incorrect number of angles: expecting {}, but {} given.'.format(
                self.num_parameters, len(angles)
            ))
        q = QuantumRegister(self._cost_operator.num_qubits, name='q')
        circuit = QuantumCircuit(q)
        if self._initial_state:
            circuit += self._initial_state.construct_circuit('circuit', q)
        else:
            circuit.u2(0, np.pi, q)
        for idx in range(self._depth):
            beta, gamma = angles[idx], angles[idx + self._depth]
            circuit += self._cost_operator.evolve(None, gamma, 'circuit', 1, quantum_registers=q)
            circuit += self._mixer_operator.evolve(None, beta, 'circuit', 1, quantum_registers=q)
        return circuit
