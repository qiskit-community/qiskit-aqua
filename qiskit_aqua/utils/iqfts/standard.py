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

from qiskit_aqua.utils.iqfts import IQFT

class Standard(IQFT):
    """A normal standard IQFT."""

    STANDARD_CONFIGURATION = {
        'name': 'STANDARD',
        'description': 'Inverse QFT',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'std_iqft_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.STANDARD_CONFIGURATION.copy())
        self._num_qubits = 0

    def init_args(self, num_qubits):
        self._num_qubits = num_qubits

    def construct_circuit(self, mode, register=None, circuit=None):
        if mode == 'vector':
            return linalg.dft(2 ** self._num_qubits, scale='sqrtn')
        elif mode == 'circuit':
            if register is None:
                register = QuantumRegister(self._num_qubits, name='q')
            if circuit is None:
                circuit = QuantumCircuit(register)

            for j in reversed(range(self._num_qubits)):
                circuit.u2(0, np.pi, register[j])
                for k in reversed(range(j)):
                    lam = -1.0 * pi / float(2 ** (j - k))
                    circuit.u1(lam / 2, register[j])
                    circuit.cx(register[j], register[k])
                    circuit.u1(-lam / 2, register[k])
                    circuit.cx(register[j], register[k])
                    circuit.u1(lam / 2, register[k])
            return circuit
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')
