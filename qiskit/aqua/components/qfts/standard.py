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

from qiskit.qasm import pi

from . import QFT
from .qft import set_up


class Standard(QFT):
    """A normal standard QFT."""

    CONFIGURATION = {
        'name': 'STANDARD',
        'description': 'QFT',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'std_qft_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits):
        super().__init__()
        self._num_qubits = num_qubits

    def construct_circuit(self, mode, qubits=None, circuit=None):
        if mode == 'vector':
            # note the difference between QFT and DFT in the phase definition:
            # QFT: \omega = exp(2*pi*i/N) ; DFT: \omega = exp(-2*pi*i/N)
            # so linalg.inv(linalg.dft()) is correct for QFT
            return linalg.inv(linalg.dft(2 ** self._num_qubits, scale='sqrtn'))
            #return linalg.dft(2 ** self._num_qubits, scale='sqrtn')
        elif mode == 'circuit':
            circuit, qubits = set_up(circuit, qubits, self._num_qubits)

            # circuit.swap(qubits[0],qubits[1])
            # for j in range(self._num_qubits):
            #     for k in range(j):
            #         lam = 1.0 * pi / float(2 ** (j-k))
            #         circuit.u1(lam / 2, qubits[j])
            #         circuit.cx(qubits[j], qubits[k])
            #         circuit.u1(-lam / 2, qubits[k])
            #         circuit.cx(qubits[j], qubits[k])
            #         circuit.u1(lam / 2, qubits[k])
            #     circuit.u2(0, np.pi, qubits[j])j

            for j in range(self._num_qubits-1,-1,-1):
                for k in range(self._num_qubits-1,j,-1):
                    lam = 1.0 * pi / float(2 ** (k-j))
                    circuit.u1(lam / 2, qubits[j])
                    circuit.cx(qubits[j], qubits[k])
                    circuit.u1(-lam / 2, qubits[k])
                    circuit.cx(qubits[j], qubits[k])
                    circuit.u1(lam / 2, qubits[k])
                circuit.u2(0, np.pi, qubits[j])

            return circuit
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')
