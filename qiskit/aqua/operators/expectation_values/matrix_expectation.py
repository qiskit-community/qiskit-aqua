# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Expectation Algorithm using Statevector simulation and matrix multiplication. """

import logging
from typing import Union
import numpy as np

from qiskit.providers import BaseBackend

from ..operator_base import OperatorBase
from .expectation_base import ExpectationBase
from ..state_functions import StateFn

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name

class MatrixExpectation(ExpectationBase):
    """ Expectation Algorithm using Statevector simulation and matrix multiplication. """

    def __init__(self,
                 operator: OperatorBase = None,
                 state: OperatorBase = None,
                 backend: BaseBackend = None) -> None:
        """
        Args:

        """
        super().__init__()
        self._operator = operator
        self._state = state
        self.backend = backend
        self._matrix_op = None

    @property
    def operator(self) -> OperatorBase:
        return self._operator

    @operator.setter
    def operator(self, operator: OperatorBase) -> None:
        self._operator = operator
        self._matrix_op = None

    @property
    def state(self) -> OperatorBase:
        """ returns state """
        return self._state

    @state.setter
    def state(self, state: OperatorBase) -> None:
        self._state = state

    def compute_expectation(self,
                            state: OperatorBase = None,
                            params: dict = None) -> Union[list, float, complex, np.ndarray]:
        # Making the matrix into a measurement allows us to handle ListOp states, dicts, etc.
        if not self._matrix_op:
            self._matrix_op = StateFn(self._operator, is_measurement=True).to_matrix_op()

        if state is None:
            state = self.state

        # If user passed in a backend, try evaluating the state on the backend.
        if self._circuit_sampler:
            state_op_mat = self._circuit_sampler.convert(state, params=params)
            return self._matrix_op.eval(state_op_mat)
        else:
            return self._matrix_op.eval(state)

    def compute_standard_deviation(self,
                                   state: OperatorBase = None,
                                   params: dict = None) -> Union[float]:
        """ compute standard deviation """
        # TODO is this right? This is what we already do today, but I'm not sure if it's correct.
        return 0.0
