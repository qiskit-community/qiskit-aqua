# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Expectation Algorithm Base """

import logging
import numpy as np
from abc import abstractmethod

from qiskit.aqua import AquaError, QuantumAlgorithm
from qiskit.aqua.operators import OpCombo, OpPrimitive

from qiskit.aqua.utils.backend_utils import (is_statevector_backend,
                                             is_aer_qasm,
                                             has_aer)

logger = logging.getLogger(__name__)


class ExpectationBase():
    """ A base for Expectation Value algorithms """

    @staticmethod
    def factory(operator, backend=None, state=None):
        """
        Args:

        """
        primitives = operator.get_primtives()
        if primitives == {'Pauli'}:
            if backend is not None and is_aer_qasm(backend):
                from .aer_pauli_expectation import AerPauliExpectation
                return AerPauliExpectation(operator=operator, backend=backend, state=state)
            elif backend is None and has_aer():
                from qiskit import Aer
                from .aer_pauli_expectation import AerPauliExpectation
                backend = Aer.get_backend('qasm_simulator')
                return AerPauliExpectation(operator=operator, backend=backend, state=state)
            else:
                from .pauli_expectation import PauliExpectation
                return PauliExpectation(operator=operator, backend=backend, state=state)
        elif primitives == {'Matrix'}:
            from .matrix_expectation import MatrixExpectation
            return MatrixExpectation(state=state, operator=operator, backend=backend)
        elif primitives == {'Instruction'}:
            from .projector_overlap import ProjectorOverlap
            return ProjectorOverlap(state=state, operator=operator, backend=backend)

    @abstractmethod
    def compute_expectation(self, state=None):
        raise NotImplementedError

    @abstractmethod
    def compute_variance(self, state=None):
        raise NotImplementedError
