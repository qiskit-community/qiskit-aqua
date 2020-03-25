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
This module contains the definition of a base class for univariate distributions.
"""

from typing import Optional, Union, List
from abc import ABC
import numpy as np

from qiskit.aqua import AquaError
from qiskit.aqua.utils.validation import validate_min
from qiskit.aqua.components.initial_states import Custom
from .uncertainty_model import UncertaintyModel


class UnivariateDistribution(UncertaintyModel, ABC):
    """
    This module contains the definition of a base class for univariate distributions.
    (Interface for discrete bounded uncertainty models assuming an equidistant grid)
    """

    def __init__(self,
                 num_target_qubits: int,
                 probabilities: Optional[Union[List[float], np.ndarray]] = None,
                 low: float = 0,
                 high: float = 1):
        r"""
        Args:
            num_target_qubits: Number of qubits it acts on,
                has a min. value of 1.
            probabilities: Probabilities for different states
            low: Lower bound, i.e., the value corresponding to \|0...0>
                (assuming an equidistant grid)
            high: Upper bound, i.e., the value corresponding to \|1...1>
                (assuming an equidistant grid)
        Raises:
            AquaError: num qubits and length of probabilities vector do not match
        """
        validate_min('num_target_qubits', num_target_qubits, 1)
        super().__init__(num_target_qubits)
        self._num_values = 2 ** self.num_target_qubits
        self._probabilities = np.array(probabilities)
        self._low = low
        self._high = high
        self._values = np.linspace(low, high, self.num_values)
        if probabilities is not None:
            if self.num_values != len(probabilities):
                raise AquaError('num qubits and length of probabilities vector do not match!')

    @property
    def low(self):
        """ returns low """
        return self._low

    @property
    def high(self):
        """ returns high """
        return self._high

    @property
    def num_values(self):
        """ returns number of values """
        return self._num_values

    @property
    def values(self):
        """ returns values """
        return self._values

    @property
    def probabilities(self):
        """ returns probabilities """
        return self._probabilities

    def build(self, qc, q, q_ancillas=None, params=None):
        """ build """
        custom_state = Custom(self.num_target_qubits, state_vector=np.sqrt(self.probabilities))
        qc.extend(custom_state.construct_circuit('circuit', q))

    @staticmethod
    def pdf_to_probabilities(pdf, low, high, num_values):
        """
        Takes a probability density function (pdf), and returns a truncated and
        discretized array of probabilities corresponding to it

        Args:
            pdf (function): probability density function
            low (float): lower bound of equidistant grid
            high (float): upper bound of equidistant grid
            num_values (int): number of grid points

        Returns:
            list: array of probabilities
        """
        probabilities = np.zeros(num_values)
        values = np.linspace(low, high, num_values)
        total = 0
        for i, _ in enumerate(values):
            probabilities[i] = pdf(values[i])
            total += probabilities[i]
        probabilities /= total
        return probabilities, values
