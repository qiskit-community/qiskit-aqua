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
The Univariate Normal Distribution.
"""

import numpy as np
from scipy.stats.distributions import norm
from .univariate_distribution import UnivariateDistribution


class NormalDistribution(UnivariateDistribution):
    """
    The Univariate Normal Distribution. It implements two approaches: 
    First, discretising bounded uncertainty models assuming an equidistant grid. 
    Second, a direct construction from Bernoulli distributed samples 
    (cf. normal_rand_float64). Both approaches require the mean and 
    standard deviation to be specified in the constructor. The second 
    approach also requires backend to be specified in the constructor.
    """

    CONFIGURATION = {
        'name': 'NormalDistribution',
        'description': 'Normal Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'NormalDistribution_schema',
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
                    'default': 3,
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_target_qubits, mu=0, sigma=1, low=-1, high=1, backend=None):
        self.validate(locals())
        probabilities, _ = UnivariateDistribution.\
            pdf_to_probabilities(lambda x: norm.pdf(x, mu, sigma), low, high, 2 ** num_target_qubits)
        super().__init__(num_target_qubits, probabilities, low, high, backend)

        # assert isinstance(mu, float) and isinstance(sigma, float)
        assert sigma > 0.0
        self._mu = mu
        self._sigma = sigma

    def normal_rand_float64(self, size: int) -> np.ndarray:
        """
        Draws a sample vector from the normal distribution with mean and variance prescribed in the constructor.
        Internally, uses the Box-Muller method and UnivariateDistribution.uniform_rand_float64. 
        """
        EPS = np.sqrt(np.finfo(np.float64).tiny)
        assert isinstance(size, int) and size > 0
        rand_vec = np.zeros((size,), dtype=np.float64)

        # Generate array of uniformly distributed samples.
        n = (3 * size) // 2
        x = np.reshape(self.uniform_rand_float64(2 * n, float(0.0), float(1.0)), (-1, 2))

        x1 = 0.0                # first sample in a pair
        c = 0                   # counter
        for d in range(size):
            r2 = 2.0
            while r2 >= 1.0 or r2 < EPS:
                # Regenerate array of uniformly distributed samples upon shortage.
                if c >= n:
                    c = 0
                    n = max(size // 10, 1)
                    x = np.reshape(self.uniform_rand_float64(2 * n, float(0.0), float(1.0)), (-1, 2))
                    # print('+++')

                x1 = 2.0 * x[c, 0] - 1.0        # first sample in a pair
                x2 = 2.0 * x[c, 1] - 1.0        # second sample in a pair
                r2 = x1 * x1 + x2 * x2
                c += 1

            f = np.sqrt(np.abs(-2.0 * np.log(r2) / r2))
            rand_vec[d] = f * x1
        return (rand_vec * self._sigma + self._mu)
