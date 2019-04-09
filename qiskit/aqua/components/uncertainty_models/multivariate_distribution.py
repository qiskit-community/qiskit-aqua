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
This module contains the definition of a base class for multivariate distributions.
"""

from abc import ABC
import numpy as np

from qiskit.aqua import Pluggable
from qiskit.aqua.components.initial_states import Custom
from .uncertainty_model import UncertaintyModel


class MultivariateDistribution(UncertaintyModel, ABC):
    """
    This module contains the definition of a base class for multivariate distributions.
    (Interface for discrete bounded uncertainty models assuming an equidistant grid)
    """

    @classmethod
    def get_section_key_name(cls):
        return Pluggable.SECTION_KEY_MULTIVARIATE_DISTRIBUTION

    def __init__(self, num_qubits, probabilities, low, high):
        """
        Constructor

        Args:
            num_qubits (:obj:`list` of :obj:`list`): assigns qubits to dimensions
            probabilities: map - maps index tuples to probabilities
            low (list): lowest value per dimension
            high (list): highest value per dimension
        """

        # derive dimension from qubit assignment
        self._dimension = len(num_qubits)
        self._num_qubits = num_qubits

        # derive total number of required qubits
        num_target_qubits = 0
        for i in range(self._dimension):
            num_target_qubits += num_qubits[i]

        # call super constructor
        super().__init__(num_target_qubits)

        # normalize probabilities
        probabilities = np.asarray(probabilities)
        probabilities = probabilities / np.sum(probabilities)

        self._num_values = []
        for i in range(self._dimension):
            self._num_values += [2 ** num_qubits[i]]
        self._probabilities = probabilities
        self._probabilities_vector = np.reshape(probabilities, 2**num_target_qubits)

        self._probabilities_vector = np.asarray(self._probabilities_vector)
        self._low = low
        self._high = high
        self._values = []
        for i in range(self._dimension):
            self._values += [np.linspace(self._low[i], self._high[i], self._num_values[i])]

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def dimension(self):
        return self._dimension

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def num_values(self):
        return self._num_values

    @property
    def values(self):
        return self._values

    @property
    def probabilities(self):
        return self._probabilities

    @property
    def probabilities_vector(self):
        return self._probabilities_vector

    def required_ancillas(self):
        return 0

    def required_ancillas_controlled(self):
        return 0

    def build(self, qc, q, q_ancillas=None, params=None):
        custom_state = Custom(self.num_target_qubits, state_vector=np.sqrt(self._probabilities_vector))
        qc.extend(custom_state.construct_circuit('circuit', q))

    @staticmethod
    def pdf_to_probabilities(pdf, low, high, num_values):
        probabilities = np.zeros(num_values)
        values = np.linspace(low, high, num_values)
        total = 0
        for i, x in enumerate(values):
            probabilities[i] = pdf(values[i])
            total += probabilities[i]
        probabilities /= total
        return probabilities, values
