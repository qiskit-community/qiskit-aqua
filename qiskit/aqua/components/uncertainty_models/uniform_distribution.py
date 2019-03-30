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
The Univariate Uniform Distribution.
"""

import numpy as np
from .univariate_distribution import UnivariateDistribution


class UniformDistribution(UnivariateDistribution):
    """
    The Univariate Uniform Distribution.
    """

    CONFIGURATION = {
        'name': 'UniformDistribution',
        'description': 'Uniform Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'UniformDistribution_schema',
            'type': 'object',
            'properties': {
                'num_target_qubits': {
                    'type': 'integer',
                    'default': 2,
                },
                'low': {
                    'type': 'number',
                    'default': 0,
                },
                'high': {
                    'type': 'number',
                    'default': 1,
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_target_qubits, low=0, high=1):
        probabilities = np.ones(2**num_target_qubits)/2**num_target_qubits
        super().__init__(num_target_qubits, probabilities, low, high)

    def required_ancillas(self):
        return 0

    def required_ancillas_controlled(self):
        return 0

    def build(self, qc, q, q_ancillas=None, params=None):
        if params is None or params['i_state'] is None:
            qc.h(q)
        else:
            for i in params['i_state']:
                qc.h(q[i])