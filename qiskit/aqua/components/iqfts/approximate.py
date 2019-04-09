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

from qiskit.aqua.circuits import FourierTransformCircuits
from . import IQFT


class Approximate(IQFT):
    """An approximate IQFT."""

    CONFIGURATION = {
        'name': 'APPROXIMATE',
        'description': 'Approximate inverse QFT',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'aiqft_schema',
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

    def _build_circuit(self, qubits=None, circuit=None, do_swaps=True):
        ftc = FourierTransformCircuits(self._num_qubits, approximation_degree=self._degree, inverse=True)
        return ftc.construct_circuit(qubits, circuit, do_swaps=do_swaps)
