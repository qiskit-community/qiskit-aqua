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

from .expectation_base import ExpectationBase

from qiskit.aqua.operators import OpVec, OpPrimitive
from qiskit.aqua.operators.converters import PauliChangeOfBasis

logger = logging.getLogger(__name__)


class PauliExpectation(ExpectationBase):
    """ An Expectation Value algorithm for taking expectations of quantum states specified by circuits over
    observables specified by Pauli Operators. Flow:

    """

    def __init__(self, operator=None, backend=None, state=None):
        """
        Args:

        """
        self._operator = operator
        self._backend = backend
        self._state = state
        self._primitives_cache = None
        self._converted_operator = None

    # TODO setters which wipe state

    def _extract_primitives(self):
        self._primitives_cache = []
        if isinstance(self._operator, OpVec):
            self._primitives_cache += [op for op in self._operator.oplist]

    def compute_expectation(self, state=None, primitives=None):
        state = state or self._state

        if not self._converted_operator:
            self._converted_operator = PauliChangeOfBasis().convert(self._operator)

        expec_op = self._converted_operator.compose(state)

        if self._primitives_cache is None:
            self._extract_primitives()
