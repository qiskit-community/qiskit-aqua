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

from qiskit_aqua.algorithms.components.feature_maps.pauli_expansion import PauliExpansion
from qiskit_aqua.algorithms.components.feature_maps import self_product


class PauliZExpansion(PauliExpansion):
    """
    Mapping data with the second order expansion followed by entangling gates.
    Refer to https://arxiv.org/pdf/1804.11326.pdf for details.

    """

    PAULI_Z_EXPANSION_CONFIGURATION = {
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

    def __init__(self, configuration=None):
        """Constructor."""
        super().__init__(configuration or self.PAULI_Z_EXPANSION_CONFIGURATION.copy())
        self._ret = {}

    def init_args(self, num_qubits, depth, entangler_map=None,
                  entanglement='full', z_order=1, data_map_func=self_product):
        """Initializer.

        Args:
            num_qubits (int): number of qubits
            depth (int): the number of repeated circuits
            entangler_map (dict): describe the connectivity of qubits
            entanglement (str): ['full', 'linear'], generate the qubit connectivitiy by predefined
                                topology
            z_order (str): z order
            data_map_func (Callable): a mapping function for data x
        """
        pauli_string = []
        for i in range(1, z_order + 1):
            pauli_string.append('Z' * i)
        super().init_args(num_qubits, depth, entangler_map, entanglement,
                          pauli_string, data_map_func)
