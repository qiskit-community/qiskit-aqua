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
The Multivariate Normal Distribution.
"""

import numpy as np
from scipy.stats import multivariate_normal
from qiskit.aqua.components.uncertainty_models.multivariate_distribution import MultivariateDistribution


class MultivariateNormalDistribution(MultivariateDistribution):
    """
    The Multivariate Normal Distribution.
    """

    CONFIGURATION = {
        'name': 'MultivariateNormalDistribution',
        'description': 'Multivariate Normal Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'MultivariateNormalDistribution_schema',
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
                    'type': 'array',
                    "items": {
                        "type": "number"
                    },
                    'default': [0.0, 0.0]
                },
                'high': {
                    'type': 'array',
                    "items": {
                        "type": "number"
                    },
                    'default': [0.12, 0.24]
                },
                'mu': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                },
                'sigma': {
                    'type': ['array', 'null'],
                    'default': None
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, low, high, mu=None, sigma=None):
        """
        Constructor.

        Circuit Factory to build a circuit that represents a multivariate normal distribution.

        Args:
            num_qubits (array or list): representing number of qubits per dimension
            low (array or list): representing lower bounds per dimension
            high (array or list): representing upper bounds per dimension
            mu (array or list): representing expected values
            sigma (array or list): representing co-variance matrix
        """
        super().validate(locals())

        if not isinstance(sigma, np.ndarray):
            sigma = np.asarray(sigma)

        if mu is None:
            mu = np.zeros(len(num_qubits))
        if sigma is None:
            sigma = np.eye(len(num_qubits))

        self.mu = mu
        self.sigma = sigma
        probs = self._compute_probabilities([], num_qubits, low, high)
        probs = np.asarray(probs) / np.sum(probs)
        super().__init__(num_qubits, probs, low, high)

    def _compute_probabilities(self, probs, num_qubits, low, high, x=None):

        for y in np.linspace(low[0], high[0], 2**num_qubits[0]):
            x_ = y if x is None else np.append(x, y)
            if len(num_qubits) == 1:
                probs.append(multivariate_normal.pdf(x_, self.mu, self.sigma))
            else:
                probs = self._compute_probabilities(probs, num_qubits[1:], low[1:], high[1:], x_)
        return probs
