# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
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

    Distribution with only two values (low, high) and the corresponding probabilities
    represented by a single qubit.
    """

    def __init__(self,
                 p: float,
                 low: float = 0,
                 high: float = 1):
        """
        Args:
            p: Probability
            low: Low value
            high: High value
        """
        probabilities = np.array([1 - p, p])
        super().__init__(1, probabilities, low, high)
        self._p = p

    @staticmethod
    def _replacement():
        return 'a 1-qubit circuit with a RY(np.arcsin(np.sqrt(p))) gate'

    @property
    def p(self):
        """ p """
        return self._p
