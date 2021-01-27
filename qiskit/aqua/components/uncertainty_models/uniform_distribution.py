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
The Univariate Uniform Distribution.
"""

import numpy as np
from qiskit.aqua.utils.validation import validate_min
from .univariate_distribution import UnivariateDistribution


class UniformDistribution(UnivariateDistribution):
    """
    The Univariate Uniform Distribution.

    Uniform distribution is defined by the number of qubits that should be used to represent the
    distribution, as well as the lower bound and upper bound of the considered interval.
    """

    def __init__(self,
                 num_target_qubits: int,
                 low: float = 0,
                 high: float = 1) -> None:
        r"""
        Args:
            num_target_qubits: Number of qubits it acts on, has a minimum value of 1.
            low: Lower bound, i.e., the value corresponding to \|0...0>
                (assuming an equidistant grid)
            high: Upper bound, i.e., the value corresponding to \|1...1>
                (assuming an equidistant grid)
        """
        validate_min('num_target_qubits', num_target_qubits, 1)
        probabilities = np.ones(2 ** num_target_qubits) / 2 ** num_target_qubits
        super().__init__(num_target_qubits, probabilities, low, high)

    @staticmethod
    def _replacement():
        return 'qiskit.circuit.library.UniformDistribution'

    def required_ancillas(self):
        return 0

    def required_ancillas_controlled(self):
        return 0

    def build(self, qc, q, q_ancillas=None, params=None):
        if params is None or params['i_state'] is None:
            for i in range(self.num_target_qubits):
                qc.h(q[i])
        else:
            for i in params['i_state']:
                qc.h(q[i])
