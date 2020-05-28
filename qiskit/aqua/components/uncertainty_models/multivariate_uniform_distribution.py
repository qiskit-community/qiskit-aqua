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
The Multivariate Uniform Distribution.
"""

from typing import Optional, List, Union
import numpy as np
from .multivariate_distribution import MultivariateDistribution


class MultivariateUniformDistribution(MultivariateDistribution):
    """
    The Multivariate Uniform Distribution.

    Although this just results in a Hadamard gate on all involved qubits, the lower and upper
    bounds and the assignment of the qubits to the different dimensions is important if used in
    a particular application.
    """

    def __init__(self,
                 num_qubits: Union[List[int], np.ndarray],
                 low: Optional[Union[List[float], np.ndarray]] = None,
                 high: Optional[Union[List[float], np.ndarray]] = None) -> None:
        """
        Args:
            num_qubits: List with the number of qubits per dimension
            low: List with the lower bounds per dimension, set to 0 for each dimension if None
            high: List with the upper bounds per dimension, set to 1 for each dimension if None
        """
        if low is None:
            low = np.zeros(num_qubits)
        if high is None:
            high = np.ones(num_qubits)

        num_values = np.prod([2**n for n in num_qubits])
        probabilities = np.ones(num_values)
        super().__init__(num_qubits, probabilities, low, high)

    def build(self, qc, q, q_ancillas=None, params=None):
        if params is None or params['i_state'] is None:
            for i in range(sum(self.num_qubits)):
                qc.h(q[i])
        else:
            for qubits in params['i_state']:
                for i in qubits:
                    qc.h(q[i])
