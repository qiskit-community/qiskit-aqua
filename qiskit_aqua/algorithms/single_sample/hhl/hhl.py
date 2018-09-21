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
The HHL algorithm.
"""

import logging

logger = logging.getLogger(__name__)


class HHL(QuantumAlgorithm):
    """The HHL algorithm."""

    HHL_CONFIGURATION = {
        'name': 'HHL',
        'description': 'The HHL Algorithm for Solving Linear Systems of equations',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'hhl_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'problems': ['energy'],
        'depends': ['eigs', 'initial_state'],
        'defaults': {
            'qpe': {
                'name': 'QPE',
                'num_ancillae': 6,
                'num_time_slices': 50,
                'expansion_mode': 'suzuki',
                'expansion_order': 2,
                'qft': {'name': 'STANDARD'}
            }
            'initial_state': {
                'name': 'ZERO'
            },
            'reciprocal':{
                'name': 'LOOKUP'
            }
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.QPE_CONFIGURATION.copy())
        self._matrix = None
        self._qpe = None
        self._ret = {}

    def init_params(self, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: EnergyInput instance
        """
        if algo_input is None:
            raise AlgorithmError("Matrix instance is required.")

        matrix = algo_input
        
        qpe_params = params.get(QuantumAlgorithm.SECTION_KEY_QPE)
        paulis_grouping = qpe_params.get(QPE.PROP_PAULIS_GROUPING)
        expansion_mode = qpe_params.get(QPE.PROP_EXPANSION_MODE)
        expansion_order = qpe_params.get(QPE.PROP_EXPANSION_ORDER)
        num_ancillae = qpe_params.get(QPE.PROP_NUM_ANCILLAE)
