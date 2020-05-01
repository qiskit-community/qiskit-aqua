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
The Univariate Bernoulli Distribution.
"""

import numpy as np
from .univariate_distribution import UnivariateDistribution

# pylint: disable=invalid-name


class BernoulliDistribution(UnivariateDistribution):
    """
    The Univariate Bernoulli Distribution.
    """

    CONFIGURATION = {
        'name': 'BernoulliDistribution',
        'description': 'Bernoulli Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'BernoulliDistribution_schema',
            'type': 'object',
            'properties': {
                'p': {
                    'type': 'number',
                    'default': 0.5,
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

    def __init__(self, p, low=0, high=1):
        probabilities = np.array([1-p, p])
        super().__init__(1, probabilities, low, high)
        self._p = p

    @property
    def p(self):
        """ p """
        return self._p
