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
        self.i_objective = i_objective

        super().__init__()

    @property
    def a_factory(self):
        return self._a_factory

    @a_factory.setter
    def a_factory(self, a_factory):
        self._a_factory = a_factory

    @property
    def q_factory(self):
        return self._q_factory

    @q_factory.setter
    def q_factory(self, q_factory):
        self._q_factory = q_factory

    def check_factories(self):
        """
        Check if a_factory has been set, and set q_factory if it hasn't been
        set already.
        """
        # check if A factory has been set
        if self._a_factory is None:
            raise AquaError("a_factory must be set!")

        # check if Q factory has been set
        if self._q_factory is None:
            self.i_objective = self.a_factory.num_target_qubits - 1
            self._q_factory = QFactory(self._a_factory, self.i_objective)
        # set i_objective if has not been set
        else:
            if self.i_objective is None:
                self.i_objective = self._q_factory.i_objective
