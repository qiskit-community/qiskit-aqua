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
The Univariate Log-Normal Distribution.
"""

from scipy.stats.distributions import lognorm
import numpy as np
from qiskit.aqua.components.uncertainty_models.univariate_distribution import UnivariateDistribution


class LogNormalDistribution(UnivariateDistribution):
    """
    The Univariate Log-Normal Distribution.
    """

    CONFIGURATION = {
        'name': 'LogNormalDistribution',
        'description': 'Log-Normal Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'LogNormalDistribution_schema',
            'type': 'object',
            'properties': {
                'num_target_qubits': {
                    'type': 'integer',
                    'default': 2,
                },
                'mu': {
                    'type': 'number',
                    'default': 0,
                },
                'sigma': {
                    'type': 'number',
                    'default': 1,
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

    def __init__(self, num_target_qubits, mu=0, sigma=1, low=0, high=1):
        """
        Constructor.

        Univariate lognormal distribution
        Args:
            num_target_qubits (int): number of qubits it acts on
            mu (float): expected value of considered normal distribution
            sigma (float): standard deviation of considered normal distribution
            low (float): lower bound, i.e., the value corresponding to |0...0>
                        (assuming an equidistant grid)
            high (float): upper bound, i.e., the value corresponding to |1...1>
                        (assuming an equidistant grid)
        """
        self.validate(locals())
        probabilities, _ = UnivariateDistribution.\
            pdf_to_probabilities(
                lambda x: lognorm.pdf(x, s=sigma, scale=np.exp(mu)),
                low, high, 2 ** num_target_qubits)
        super().__init__(num_target_qubits, probabilities, low, high)
