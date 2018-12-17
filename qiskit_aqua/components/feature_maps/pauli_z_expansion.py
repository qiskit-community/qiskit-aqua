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
"""
This module contains the definition of a base class for
feature map. Several types of commonly used approaches.
"""

from qiskit_aqua.components.feature_maps import PauliExpansion, self_product


class PauliZExpansion(PauliExpansion):
    """
    Mapping data with the second order expansion followed by entangling gates.

    Refer to https://arxiv.org/pdf/1804.11326.pdf for details.
    """

    CONFIGURATION = {
        'name': 'PauliZExpansion',
        'description': 'Pauli Z expansion for feature map (any order)',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'Pauli_Z_Expansion_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
                'entangler_map': {
                    'type': ['object', 'null'],
                    'default': None
                },
                'entanglement': {
                    'type': 'string',
                    'default': 'full',
                    'oneOf': [
                        {'enum': ['full', 'linear']}
                    ]
                },
                'z_order': {
                    'type': 'integer',
                    'minimum': 1,
                    'default': 2
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, depth=2, entangler_map=None,
                 entanglement='full', z_order=2, data_map_func=self_product):
        """Constructor."""
        self.validate(locals())
        pauli_string = []
        for i in range(1, z_order + 1):
            pauli_string.append('Z' * i)
        super().__init__(num_qubits, depth, entangler_map, entanglement,
                         paulis=pauli_string, data_map_func=data_map_func)
