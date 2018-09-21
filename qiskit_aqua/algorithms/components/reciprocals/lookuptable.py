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

from qiskit_aqua.algorithms.components.reciprocals import Reciprocal

class LookupTable(Reciprocal):
    """An approximated lookup table method for calculating the reciprocal."""

    STANDARD_CONFIGURATION = {
        'name': 'LOOKUP',
        'description': 'Approximate LookupTable',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'reciprocal_lookup_schema',
            'type': 'object',
            'properties': {
                'error': {
                    'type': 'number',
                    'default': 0.05
                },
                'C': {
                    'type': 'number',
                    'default': 1
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.STANDARD_CONFIGURATION.copy())
        self._error = None
        self._C = None
        self._ancilla_register = None
        self._negative_evals = False

    def init_args(self, error=None, C=None, negative_evals=False):
        self._error = error
        self._C = C
        self._negative_evals = negative_evals

    def construct_circuit(self, mode, eigenvalue_register=None, evo_time=1):
        if mode == 'vector':
            raise ValueError('mode vector not yet supported.')
        elif mode == 'circuit':
            q = eigenvalue_register
            a = QuantumRegister(1)
            self._ancilla_register = a
            qc = QuantumCircuit(q, a)
            return qc
