# -*- coding: utf-8 -*-

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
The Multivariate Log-Normal Distribution.
"""

from typing import Optional, List, Union
import numpy as np
from scipy.stats import multivariate_normal
from .multivariate_distribution import MultivariateDistribution


class MultivariateLogNormalDistribution(MultivariateDistribution):
    """
    The Multivariate Log-Normal Distribution.
    """

    # pylint: disable=invalid-name
    def __init__(self,
                 num_qubits: Union[List[int], np.ndarray],
                 low: Optional[Union[List[float], np.ndarray]] = None,
                 high: Optional[Union[List[float], np.ndarray]] = None,
                 mu: Optional[Union[List[float], np.ndarray]] = None,
                 cov: Optional[Union[List[float], np.ndarray]] = None) -> None:
        """
        Args:
            num_qubits: Number of qubits per dimension
            low: Lower bounds per dimension
            high: Upper bounds per dimension
            mu: Expected values
            cov: Co-variance matrix
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
        super().__init__(num_qubits, probs, low, high)
        self._values = values

    def _compute_probabilities(self, probs, values, num_qubits, low, high, x=None):

        for y in np.linspace(low[0], high[0], 2 ** num_qubits[0]):
            x__ = y if x is None else np.append(x, y)
            if len(num_qubits) == 1:
                # map probabilities from normal to log-normal
                # reference:
                # https://stats.stackexchange.com/questions/214997/multivariate-log-normal-probabiltiy-density-function-pdf
                if np.min(x__) > 0.0:
                    phi_x_ = np.log(x__)
                    det_j_phi = 1 / np.prod(x__)
                    prob = multivariate_normal.pdf(phi_x_, mean=self.mu, cov=self.cov) * det_j_phi
                    probs.append(prob)
                else:
                    probs.append(0.0)
                values.append(x__)
            else:
                probs, values = self._compute_probabilities(probs,
                                                            values,
                                                            num_qubits[1:], low[1:], high[1:], x__)
        return probs, values
