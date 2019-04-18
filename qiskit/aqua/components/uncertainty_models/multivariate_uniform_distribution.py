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
The Multivariate Uniform Distribution.
"""

import numpy as np
from qiskit.aqua.components.uncertainty_models.multivariate_distribution import MultivariateDistribution


class MultivariateUniformDistribution(MultivariateDistribution):
    """
    The Multivariate Uniform Distribution.
    """

    CONFIGURATION = {
        'name': 'MultivariateUniformDistribution',
        'description': 'Multivariate Uniform Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'MultivariateUniformDistribution_schema',
            'type': 'object',
            'properties': {
                'num_qubits': {
                    'type': 'array',
                    "items": {
                        "type": "number"
                    },
                    'default': [2, 2]
                },
                'low': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                },
                'high': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, low=None, high=None):
        """
        Multivariate uniform distribution
        Args:
            num_qubits (array or list): list with the number of qubits per dimension
            low (array or list): list with the lower bounds per dimension, set to 0 for each dimension if None
            high (array or list): list with the upper bounds per dimension, set to 1 for each dimension if None
        """
        super().validate(locals())

        if low is None:
            low = np.zeros(num_qubits)
        if high is None:
            high = np.ones(num_qubits)

        num_values = np.prod([2**n for n in num_qubits])
        probabilities = np.ones(num_values)
        super().__init__(num_qubits, low, high, probabilities)

    def build(self, qc, q, q_ancillas=None, params=None):
        if params is None or params['i_state'] is None:
            for i in range(sum(self.num_qubits)):
                qc.h(q[i])
        else:
            for qubits in params['i_state']:
                for i in qubits:
                    qc.h(q[i])
