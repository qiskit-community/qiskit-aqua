# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
The Multivariate Normal Distribution.
"""

import numpy as np
from scipy.stats import multivariate_normal
from qiskit.aqua.components.uncertainty_models.multivariate_distribution \
    import MultivariateDistribution

# pylint: disable=invalid-name


class MultivariateNormalDistribution(MultivariateDistribution):
    """
    The Multivariate Normal Distribution.
    """

    CONFIGURATION = {
        'name': 'MultivariateNormalDistribution',
        'description': 'Multivariate Normal Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
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

    def __init__(self, num_qubits, low=None, high=None, mu=None, sigma=None):
        """
        Constructor.

        Circuit Factory to build a circuit that represents a multivariate normal distribution.

        Args:
            num_qubits (Union(list, numpy.ndarray)): representing number of qubits per dimension
            low (Union(list, numpy.ndarray)): representing lower bounds per dimension
            high (Union(list, numpy.ndarray)): representing upper bounds per dimension
            mu (Union(list, numpy.ndarray)): representing expected values
            sigma (Union(list, numpy.ndarray)): representing co-variance matrix
        """
        super().validate(locals())

        if not isinstance(sigma, np.ndarray):
            sigma = np.asarray(sigma)

        dimension = len(num_qubits)

        if mu is None:
            mu = np.zeros(dimension)
        if sigma is None:
            sigma = np.eye(dimension)
        if low is None:
            low = -np.ones(dimension)
        if high is None:
            high = np.ones(dimension)

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
