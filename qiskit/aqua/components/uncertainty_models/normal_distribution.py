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
The Univariate Normal Distribution.
"""

from scipy.stats.distributions import norm
from qiskit.aqua.utils.validation import validate_min
from .univariate_distribution import UnivariateDistribution


class NormalDistribution(UnivariateDistribution):
    """
    The Univariate Normal Distribution.

    Normal distribution, truncated to lower and upper bound and discretized on a grid defined by
    the number of qubits.
    """

    def __init__(self,
                 num_target_qubits: int,
                 mu: float = 0,
                 sigma: float = 1,
                 low: float = -1,
                 high: float = 1) -> None:
        r"""
        Args:
            num_target_qubits: Number of qubits it acts on, has a minimum value of 1.
            mu: Expected value of considered normal distribution
            sigma: standard deviation of considered normal distribution
            low: Lower bound, i.e., the value corresponding to \|0...0>
                (assuming an equidistant grid)
            high: Upper bound, i.e., the value corresponding to \|1...1>
                (assuming an equidistant grid)
        """
        validate_min('num_target_qubits', num_target_qubits, 1)
        probabilities, _ = UnivariateDistribution.\
            pdf_to_probabilities(
                lambda x: norm.pdf(x, mu, sigma), low, high, 2 ** num_target_qubits)
        super().__init__(num_target_qubits, probabilities, low, high)
