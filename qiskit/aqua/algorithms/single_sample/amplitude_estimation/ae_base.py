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
The Amplitude Estimation Algorithm.
"""

import logging
from abc import abstractmethod

from qiskit.aqua import AquaError
from qiskit.aqua.algorithms import QuantumAlgorithm

from .q_factory import QFactory

logger = logging.getLogger(__name__)


class AmplitudeEstimationBase(QuantumAlgorithm):
    """
    The Amplitude Estimation Base class.
    """

    @abstractmethod
    def __init__(self, a_factory=None, q_factory=None, i_objective=None):
        self._a_factory = a_factory
        self._q_factory = q_factory
        self._i_objective = i_objective

        super().__init__()

    @property
    def a_factory(self):
        """ returns a factory """
        return self._a_factory

    @a_factory.setter
    def a_factory(self, a_factory):
        """ sets a factory """
        self._a_factory = a_factory

    @property
    def q_factory(self):
        if self._q_factory is not None:
            return self._q_factory

        if self._a_factory is not None:
            return QFactory(self._a_factory, self.i_objective)

        return None

    @q_factory.setter
    def q_factory(self, q_factory):
        """ sets q factory """
        self._q_factory = q_factory

    @property
    def i_objective(self):
        if self._i_objective is not None:
            return self._i_objective

        if self._q_factory is not None:
            return self._q_factory.i_objective

        return self.a_factory.num_target_qubits - 1

    def check_factories(self):
        """
        Check if a_factory has been set, and set q_factory if it hasn't been
        set already.
        """
        # check if A factory has been set
        if self._a_factory is None:
            raise AquaError("a_factory must be set!")
