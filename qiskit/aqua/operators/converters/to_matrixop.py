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

""" Expectation Algorithm Base """

import logging

from ..operator_base import OperatorBase
from ..operator_primitives import OpPrimitive
from ..operator_combos import OpVec
from ..state_functions import StateFn, StateFnOperator
from .converter_base import ConverterBase

logger = logging.getLogger(__name__)


class ToMatrixOp(ConverterBase):
    """ Expectation Algorithm Base """
    def __init__(self, traverse=True):
        self._traverse = traverse

    def convert(self, operator):

        # TODO: Fix this
        if isinstance(operator, OpVec):
            return operator.__class__(operator.traverse(self.convert), coeff=operator.coeff)
        elif isinstance(operator, StateFnOperator):
            return StateFnOperator(OpPrimitive(operator.to_density_matrix()),
                                   is_measurement=operator.is_measurement)
        elif isinstance(operator, StateFn):
            return StateFn(operator.to_matrix(), is_measurement=operator.is_measurement)
        elif isinstance(operator, OperatorBase):
            return OpPrimitive(operator.to_matrix())
        else:
            raise TypeError('Cannot convert type {} to OpMatrix'.format(type(operator)))
