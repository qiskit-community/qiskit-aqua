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
The Multivariate Normal Distribution.
"""

from typing import Optional, List, Union
import numpy as np
from scipy.stats import multivariate_normal
from .multivariate_distribution import MultivariateDistribution

# pylint: disable=invalid-name


class MultivariateNormalDistribution(MultivariateDistribution):
    """
    The Multivariate Normal Distribution.

    Provides a discretized and truncated normal distribution loaded into a quantum state.
    Truncation bounds are given by lower and upper bound and discretization is specified by the
    number of qubits per dimension.
    """

    def __init__(self,
                 num_qubits: Union[List[int], np.ndarray],
                 low: Optional[Union[List[float], np.ndarray]] = None,
                 high: Optional[Union[List[float], np.ndarray]] = None,
                 mu: Optional[Union[List[float], np.ndarray]] = None,
                 sigma: Optional[Union[List[float], np.ndarray]] = None) -> None:
        """
        Args:
            num_qubits: Number of qubits per dimension
            low: Lower bounds per dimension
            high: Upper bounds per dimension
            mu: Expected values
            sigma: Co-variance matrix
        """
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

    @staticmethod
    def _replacement():
        return 'qiskit.circuit.library.NormalDistribution'

    def _compute_probabilities(self, probs, num_qubits, low, high, x=None):

        for y in np.linspace(low[0], high[0], 2**num_qubits[0]):
            x_ = y if x is None else np.append(x, y)
            if len(num_qubits) == 1:
                probs.append(multivariate_normal.pdf(x_, self.mu, self.sigma))
            else:
                probs = self._compute_probabilities(probs, num_qubits[1:], low[1:], high[1:], x_)
        return probs
