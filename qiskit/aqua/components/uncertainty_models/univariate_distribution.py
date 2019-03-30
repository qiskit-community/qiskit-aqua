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
This module contains the definition of a base class for univariate distributions.
"""

from abc import ABC
import numpy as np

from qiskit.aqua import AquaError, Pluggable
from qiskit.aqua.components.initial_states import Custom
from .uncertainty_model import UncertaintyModel


class UnivariateDistribution(UncertaintyModel, ABC):
    """
    This module contains the definition of a base class for univariate distributions.
    (Interface for discrete bounded uncertainty models assuming an equidistant grid)
    """

    @classmethod
    def get_section_key_name(cls):
        return Pluggable.SECTION_KEY_UNIVARIATE_DISTRIBUTION

    def __init__(self, num_target_qubits, probabilities, low=0, high=1):
        super().__init__(num_target_qubits)
        self._num_values = 2 ** self.num_target_qubits
        self._probabilities = np.array(probabilities)
        self._low = low
        self._high = high
        self._values = np.linspace(low, high, self.num_values)
        if self.num_values != len(probabilities):
            raise AquaError('num qubits and length of probabilities vector do not match!')

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

    def required_ancillas(self):
        return 0

    def required_ancillas_controlled(self):
        return 0

    def build(self, qc, q, q_ancillas=None, params=None):
        custom_state = Custom(self.num_target_qubits, state_vector=np.sqrt(self.probabilities))
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
