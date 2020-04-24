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

from ..operator_base import OperatorBase
from .expectation_base import ExpectationBase
from ..list_ops import ListOp, ComposedOp
from ..state_fns.operator_state_fn import OperatorStateFn

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name

class MatrixExpectation(ExpectationBase):
    """ Expectation Algorithm using Statevector simulation and matrix multiplication. """

    def convert(self, operator: OperatorBase) -> OperatorBase:
        """ Accept an Operator and return a new Operator with the Pauli measurements replaced by
        Matrix based measurements. """
        if isinstance(operator, OperatorStateFn) and operator.is_measurement:
            return operator.to_matrix_op()
        elif isinstance(operator, ListOp):
            return operator.traverse(self.convert)
        else:
            return operator

    def compute_variance(self, exp_op: OperatorBase) -> Union[list, float]:
        """ compute variance """

        # Need to do this to mimic Op structure
        def sum_variance(operator):
            if isinstance(operator, ComposedOp):
                return 0.0
            elif isinstance(operator, ListOp):
                return operator._combo_fn([sum_variance(op) for op in operator.oplist])
            else:
                return 0.0

        return sum_variance(exp_op)
