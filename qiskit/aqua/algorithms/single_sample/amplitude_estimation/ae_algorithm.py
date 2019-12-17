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

from qiskit.aqua.algorithms import QuantumAlgorithm

from .q_factory import QFactory

logger = logging.getLogger(__name__)


class AmplitudeEstimationAlgorithm(QuantumAlgorithm):
    """
    The Quantum Amplitude Estimation (QAE) algorithm base class.

    In general, QAE algorithms aim to approximate the amplitude of a certain, marked state.
    This amplitude is encoded in the so-called A operator, performing the mapping

            A |0>_n |0> = sqrt{1 - a} |psi_0>_n |0> + sqrt{a} |psi_1>_n |1>

    where the amplitude `a` (in [0, 1]) is approximated, and |psi_0> and |psi_1> are two
    normalized, not necessarily orthogonal, states.
    In the QAE algorithms, the Grover operator Q is used, which is defined as

            Q = -A S_0 A^{-1} S_psi0,

    where S_0 reflects about the |0>_n state and S_psi0 reflects about |psi_0>_n.

    See https://arxiv.org/abs/quant-ph/0005055 for more detail about QAE.
    """

    @abstractmethod
    def __init__(self, a_factory=None, q_factory=None, i_objective=None):
        self._a_factory = a_factory
        self._q_factory = q_factory
        self._i_objective = i_objective

        super().__init__()

    @property
    def a_factory(self):
        """
        Get the A operator encoding the amplitude `a` that's approximated, i.e.

            A |0>_n |0> = sqrt{1 - a} |psi_0>_n |0> + sqrt{a} |psi_1>_n |1>

        see the original Brassard paper (https://arxiv.org/abs/quant-ph/0005055) for more detail.

        Returns:
            CircuitFactory: the A operator as CircuitFactory
        """
        return self._a_factory

    @a_factory.setter
    def a_factory(self, a_factory):
        """
        Set the A operator, that encodes the amplitude to be estimated.

        Args:
            a_factory (CircuitFactory): the A Operator
        """
        self._a_factory = a_factory

    @property
    def q_factory(self):
        """
        Get the Q operator, or Grover-operator for the Amplitude Estimation algorithm, i.e.

            Q = -A S_0 A^{-1} S_psi0,

        where S_0 reflects about the |0>_n state and S_psi0 reflects about |psi_0>_n.
        See https://arxiv.org/abs/quant-ph/0005055 for more detail.

        If the Q operator is not set, we try to build it from the A operator.
        If neither the A operator is set, None is returned.

        Returns:
            QFactory: returns the current Q factory of the algorithm
        """
        if self._q_factory is not None:
            return self._q_factory

        if self._a_factory is not None:
            return QFactory(self._a_factory, self.i_objective)

        return None

    @q_factory.setter
    def q_factory(self, q_factory):
        """
        Set the Q operator as QFactory.

        Args:
            q_factory (QFactory): the specialized Q operator
        """
        self._q_factory = q_factory

    @property
    def i_objective(self):
        """
        Get the index of the objective qubit. The objective qubit marks the |psi_0> state (called
        'bad states' in https://arxiv.org/abs/quant-ph/0005055)
        with |0> and |psi_1> ('good' states) with |1>.
        If the A operator performs the mapping

            A |0>_n |0> = sqrt{1 - a} |psi_0>_n |0> + sqrt{a} |psi_1>_n |1>

        then, the objective qubit is the last one (which is either |0> or |1>).

        If the objective qubit (i_objective) is not set, we check if the Q operator (q_factory) is
        set and return the index specified there. If the q_factory is not defined,
        the index equals the number of qubits of the A operator (a_factory) minus one.
        If also the a_factory is not set, return None.

        Returns:
            int: the index of the objective qubit
        """
        if self._i_objective is not None:
            return self._i_objective

        if self._q_factory is not None:
            return self._q_factory.i_objective

        if self._a_factory is not None:
            return self.a_factory.num_target_qubits - 1

        return None
