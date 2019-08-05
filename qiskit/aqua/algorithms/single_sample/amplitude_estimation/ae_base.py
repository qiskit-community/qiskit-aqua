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

        if q_factory is not None and i_objective is None:
            raise AquaError('i_objective must be set for custom q_factory')

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
    def q_factory(self, q_factory_and_i_objective):
        """
        Setter using
            ae.q_factory = (q_factory, i_objective)
        """
        try:
            self._q_factory, self.i_objective = q_factory_and_i_objective
        except ValueError:
            raise ValueError("Pass an iterable (q_factory, i_objective)")

    def set_q_factory(self, q_factory, i_objective):
        """
        Oldschool setter using
            ae.set_q_factory(q_factory, i_objective)
        """
        if i_objective is None:
            raise AquaError('i_objective must be set for custom q_factory')
        self.q_factory = q_factory
        self.i_objective = i_objective

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

        # check if i_objective has been set if a custom Q factory is used
        elif self.i_objective is None:
            raise AquaError('i_objective must be set for custom q_factory')
