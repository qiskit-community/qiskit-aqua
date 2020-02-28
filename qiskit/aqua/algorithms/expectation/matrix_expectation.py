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
from qiskit.aqua.operators import OpMatrix, StateFn

logger = logging.getLogger(__name__)


class MatrixExpectation(ExpectationBase):
    """ A base for Expectation Value algorithms """

    def __init__(self, operator=None, state=None):
        """
        Args:

        """
        super().__init__()
        self._operator = operator
        self._state = state
        self._matrix_op = None

    def compute_expectation(self, state=None):
        # Making the matrix into a measurement allows us to handle OpVec states, dicts, etc.
        if state or not self._reduced_expect_op:
            self._matrix_op = StateFn(OpMatrix(self._operator.to_matrix()), is_measurement=True)
            # TODO to_quantum_runnable converter?

        return self._matrix_op.eval(state)
