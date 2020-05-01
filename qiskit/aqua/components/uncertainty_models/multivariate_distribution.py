# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
        return Pluggable.SECTION_KEY_MULTIVARIATE_DIST

    def __init__(self, num_qubits, probabilities=None, low=None, high=None):
        """
        Constructor.

        Args:
            num_qubits (Union(list, numpy.ndarray)): assigns qubits to dimensions
            probabilities (map): map - maps index tuples to probabilities
            low (Union(list, numpy.ndarray)): lowest value per dimension
            high (Union(list, numpy.ndarray)): highest value per dimension
        """

        self._values = 0
        # derive dimension from qubit assignment
        self._dimension = len(num_qubits)
        self._num_qubits = num_qubits

        # derive total number of required qubits
        num_target_qubits = 0
        for i in range(self._dimension):
            num_target_qubits += num_qubits[i]

        # call super constructor
        super().__init__(num_target_qubits)

        self._num_values = []
        for i in range(self._dimension):
            self._num_values += [2 ** num_qubits[i]]

        if probabilities is not None:

            # normalize probabilities
            probabilities = np.asarray(probabilities)
            probabilities = probabilities / np.sum(probabilities)

            self._probabilities = probabilities
            self._probabilities_vector = np.reshape(probabilities, 2**num_target_qubits)
            self._probabilities_vector = np.asarray(self._probabilities_vector)
        else:
            self._probabilities = None
            self._probabilities_vector = None

        if low is not None:
            self._low = low
        else:
            self._low = np.zeros(self.dimension)

        if high is not None:
            self._high = high
        else:
            self._high = np.zeros(self.dimension)
            for i in range(self.dimension):
                self._high[i] = 2**num_qubits[i] - 1

    @property
    def num_qubits(self):
        """ returns num qubits """
        return self._num_qubits

    @property
    def dimension(self):
        """ returns dimensions """
        return self._dimension

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

    @property
    def probabilities_vector(self):
        """ returns probabilities vector """
        return self._probabilities_vector

    def build(self, qc, q, q_ancillas=None, params=None):
        """ build """
        custom_state = Custom(self.num_target_qubits,
                              state_vector=np.sqrt(self._probabilities_vector))
        qc.extend(custom_state.construct_circuit('circuit', q))

    @staticmethod
    def pdf_to_probabilities(pdf, low, high, num_values):
        """ pdf to probabilities """
        probabilities = np.zeros(num_values)
        values = np.linspace(low, high, num_values)
        total = 0
        for i, _ in enumerate(values):
            probabilities[i] = pdf(values[i])
            total += probabilities[i]
        probabilities /= total
        return probabilities, values
