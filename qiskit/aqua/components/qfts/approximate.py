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

from scipy import linalg
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.qasm import pi

from qiskit.aqua.components.qfts import QFT


class Approximate(QFT):
    """An approximate QFT."""

    CONFIGURATION = {
        'name': 'APPROXIMATE',
        'description': 'Approximate QFT',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'aqft_schema',
            'type': 'object',
            'properties': {
                'degree': {
                    'type': 'integer',
                    'default': 0,
                    'minimum': 0
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, degree=0):
        self.validate(locals())
        super().__init__()
        self._num_qubits = num_qubits
        self._degree = degree

    def construct_circuit(self, mode, register=None, circuit=None):
        if mode == 'vector':
            raise NotImplementedError()
        elif mode == 'circuit':
            if register is None:
                register = QuantumRegister(self._num_qubits, name='q')
            if circuit is None:
                circuit = QuantumCircuit()
            if not circuit.has_register(register):
                circuit.add_register(register)

            for j in range(self._num_qubits):
                # neighbor_range = range(np.max([0, j - self._degree + 1]), j)
                neighbor_range = range(np.max([0, j - self._num_qubits + self._degree + 1]), j)
                for k in neighbor_range:
                    lam = 1.0 * pi / float(2 ** (j - k))
                    circuit.u1(lam / 2, register[j])
                    circuit.cx(register[j], register[k])
                    circuit.u1(-lam / 2, register[k])
                    circuit.cx(register[j], register[k])
                    circuit.u1(lam / 2, register[k])
                circuit.u2(0, np.pi, register[j])
            return circuit
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')
