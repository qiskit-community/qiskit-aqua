# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
The Gaussian Conditional Independence Model for Credit Risk
Reference: https://arxiv.org/abs/1412.1183
Dependency between individual risk variables and latent variable is approximated linearly.
"""

from typing import Optional, List, Union
import numpy as np
from scipy.stats.distributions import norm

from qiskit.circuit.library import LinearPauliRotations
from .multivariate_distribution import MultivariateDistribution
from .normal_distribution import NormalDistribution

# pylint: disable=invalid-name


class GaussianConditionalIndependenceModel(MultivariateDistribution):
    """The Gaussian Conditional Independence Model for Credit Risk.

    Reference: https://arxiv.org/abs/1412.1183

    Dependency between individual risk variables and latent variable is approximated linearly.
    """

    def __init__(self,
                 n_normal: int,
                 normal_max_value: float,
                 p_zeros: Union[List[float], np.ndarray],
                 rhos: Union[List[float], np.ndarray],
                 i_normal: Optional[Union[List[float], np.ndarray]] = None,
                 i_ps: Optional[Union[List[float], np.ndarray]] = None) -> None:
        """
        Args:
            n_normal: Number of qubits to represent the latent normal random variable Z
            normal_max_value: Min/max value to truncate the latent normal random variable Z
            p_zeros: Standard default probabilities for each asset
            rhos: Sensitivities of default probability of assets with respect to latent variable Z
            i_normal: Indices of qubits to represent normal variable
            i_ps: Indices of qubits to represent asset defaults
        """
        self.n_normal = n_normal
        self.normal_max_value = normal_max_value
        self.p_zeros = p_zeros
        self.rhos = rhos
        self.K = len(p_zeros)
        num_qubits = [n_normal] + [1] * self.K

        # set and store indices
        if i_normal is not None:
            self.i_normal = i_normal
        else:
            self.i_normal = list(range(n_normal))

        if i_ps is not None:
            self.i_ps = i_ps
        else:
            self.i_ps = list(range(n_normal, n_normal + self.K))

        # get normal (inverse) CDF and pdf
        def F(x):
            return norm.cdf(x)

        def F_inv(x):
            return norm.ppf(x)

        def f(x):
            return norm.pdf(x)

        # set low/high values
        low = [-normal_max_value] + [0] * self.K
        high = [normal_max_value] + [1] * self.K

        # call super constructor
        super().__init__(num_qubits, low=low, high=high)

        # create normal distribution
        self._normal = NormalDistribution(n_normal, 0, 1, -normal_max_value, normal_max_value)

        # create linear rotations for conditional defaults
        self._slopes = np.zeros(self.K)
        self._offsets = np.zeros(self.K)
        for k in range(self.K):

            psi = F_inv(p_zeros[k]) / np.sqrt(1 - rhos[k])

            # compute slope / offset
            slope = -np.sqrt(rhos[k]) / np.sqrt(1 - rhos[k])
            slope *= f(psi) / np.sqrt(1 - F(psi)) / np.sqrt(F(psi))
            offset = 2 * np.arcsin(np.sqrt(F(psi)))

            # adjust for integer to normal range mapping
            offset += slope * (-normal_max_value)
            slope *= 2 * normal_max_value / (2 ** n_normal - 1)

            self._offsets[k] = offset
            self._slopes[k] = slope

    @staticmethod
    def _replacement():
        return 'qiskit.finance.applications.GaussianConditionalIndependenceModel'

    def build(self, qc, q, q_ancillas=None, params=None):
        self._normal.build(qc, q, q_ancillas)
        for k in range(self.K):
            lry = LinearPauliRotations(self.n_normal, self._slopes[k], self._offsets[k])
            qc.append(lry.to_instruction(), self.i_normal + [self.i_ps[k]])
