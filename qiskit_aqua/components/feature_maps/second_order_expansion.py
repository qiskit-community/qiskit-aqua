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

from qiskit_aqua.components.feature_maps import PauliZExpansion, self_product


class SecondOrderExpansion(PauliZExpansion):
    """
    Mapping data with the second order expansion followed by entangling gates.

    Refer to https://arxiv.org/pdf/1804.11326.pdf for details.
    """

    CONFIGURATION = {
        'name': 'SecondOrderExpansion',
        'description': 'Second order expansion for feature map',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'Second_Order_Expansion_schema',
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
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, depth=2, entangler_map=None,
                 entanglement='full', data_map_func=self_product):
        """Constructor.

        Args:
            num_qubits (int): number of qubits
            depth (int): the number of repeated circuits
            entangler_map (dict): describe the connectivity of qubits
            entanglement (str): ['full', 'linear'], generate the qubit connectivitiy by predefined
                                topology
            data_map_func (Callable): a mapping function for data x
        """
        self.validate(locals())
        super().__init__(num_qubits, depth, entangler_map, entanglement,
                         z_order=2, data_map_func=data_map_func)
