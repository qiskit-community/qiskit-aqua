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
The Multivariate Log-Normal Distribution.
"""

import numpy as np
from scipy.stats import multivariate_normal
from qiskit.aqua.components.uncertainty_models.multivariate_distribution import MultivariateDistribution


class MultivariateLogNormalDistribution(MultivariateDistribution):
    """
    The Multivariate Log-Normal Distribution.
    """

    CONFIGURATION = {
        'name': 'MultivariateLogNormalDistribution',
        'description': 'Multivariate Log-Normal Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'MultivariateLogNormalDistribution_schema',
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
                'cov': {
                    'type': ['array', 'null'],
                    'default': None
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, low=None, high=None, mu=None, cov=None):
        """
        Constructor.

        Circuit Factory to build a circuit that represents a multivariate log-normal distribution.

        Args:
            num_qubits (array or list): representing number of qubits per dimension
            low (array or list): representing lower bounds per dimension
            high (array or list): representing upper bounds per dimension
            mu (array or list): representing expected values
            cov (array or list): representing co-variance matrix
        """

        dimension = len(num_qubits)
        if mu is None:
            mu = np.zeros(dimension)
        if cov is None:
            cov = np.eye(dimension)
        if low is None:
            low = np.zeros(dimension)
        if high is None:
            high = np.ones(dimension)

        self.mu = mu
        self.cov = cov
        probs, values = self._compute_probabilities([], [], num_qubits, low, high)
        probs = np.asarray(probs) / np.sum(probs)
        self._values = values
        super().__init__(num_qubits, low, high, probs)

    def _compute_probabilities(self, probs, values, num_qubits, low, high, x=None):

        for y in np.linspace(low[0], high[0], 2 ** num_qubits[0]):
            x_ = y if x is None else np.append(x, y)
            if len(num_qubits) == 1:
                # map probabilities from normal to log-normal
                # reference:
                # https://stats.stackexchange.com/questions/214997/multivariate-log-normal-probabiltiy-density-function-pdf
                if np.min(x_) > 0.0:
                    phi_x_ = np.log(x_)
                    det_J_phi = 1 / np.prod(x_)
                    prob = multivariate_normal.pdf(phi_x_, mean=self.mu, cov=self.cov) * det_J_phi
                    probs.append(prob)
                else:
                    probs.append(0.0)
                values.append(x_)
            else:
                probs, values = self._compute_probabilities(probs, values, num_qubits[1:], low[1:], high[1:], x_)
        return probs, values
