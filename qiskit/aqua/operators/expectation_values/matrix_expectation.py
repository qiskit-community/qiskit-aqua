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

from qiskit import BasicAer

from .expectation_base import ExpectationBase
from qiskit.aqua.operators import OpMatrix, StateFn, OpVec
from qiskit.aqua.operators.converters import ToMatrixOp

logger = logging.getLogger(__name__)


class MatrixExpectation(ExpectationBase):
    """ A base for Expectation Value algorithms """

    def __init__(self, operator=None, backend=None, state=None):
        """
        Args:

        """
        super().__init__()
        self._operator = operator
        self._state = state
        if backend is not None:
            self.set_backend(backend)
        self._matrix_op = None

    @property
    def operator(self):
        return self._operator

    @operator.setter
    def operator(self, operator):
        self._operator = operator
        self._matrix_op = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    def compute_expectation(self, state=None, params=None):
        # Making the matrix into a measurement allows us to handle OpVec states, dicts, etc.
        if not self._matrix_op:
            mat_conversion = self._operator.to_matrix()
            if isinstance(mat_conversion, list):
                def recursive_opvec(t):
                    if isinstance(t, list):
                        return OpVec([recursive_opvec(t_op) for t_op in t])
                    else:
                        return StateFn(OpMatrix(t), is_measurement=True)
                self._matrix_op = recursive_opvec(mat_conversion)
            else:
                self._matrix_op = StateFn(OpMatrix(mat_conversion), is_measurement=True)

            # TODO: switch to this
            # self._matrix_op = ToMatrixOp().convert(self._operator)

        if state is None:
            state = self.state

        if self._circuit_sampler:
            state_op_mat = self._circuit_sampler.convert(state, params=params)
            return self._matrix_op.eval(state_op_mat)
        else:
            return self._matrix_op.eval(state)

    def compute_standard_deviation(self):
        # TODO is this right? This is what we already do today, but I'm not sure if it's correct.
        return 0.0
