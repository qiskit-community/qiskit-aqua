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
The Univariate Bernoulli Distribution.
"""

import numpy as np
from .univariate_distribution import UnivariateDistribution


class BernoulliDistribution(UnivariateDistribution):
    """
    The Univariate Bernoulli Distribution.
    """

    CONFIGURATION = {
        'name': 'BernoulliDistribution',
        'description': 'Bernoulli Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
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
        return self._p
