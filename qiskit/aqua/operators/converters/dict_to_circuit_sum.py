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
from ..state_functions import DictStateFn, VectorStateFn, CircuitStateFn
from ..combo_operators import ListOp
from .converter_base import ConverterBase

logger = logging.getLogger(__name__)


class DictToCircuitSum(ConverterBase):
    """ Very naively convert DictStateFns to sums of CircuitStateFns which each
    prepare the bit strings in the keys of the dict."""

    def __init__(self,
                 traverse: bool = True,
                 convert_dicts: bool = True,
                 convert_vectors: bool = True):
        self._traverse = traverse
        self._convert_dicts = convert_dicts
        self._convert_vectors = convert_vectors

    def convert(self, operator: OperatorBase) -> OperatorBase:

        if isinstance(operator, DictStateFn) and self._convert_dicts:
            return CircuitStateFn.from_dict(operator.primitive)
        if isinstance(operator, VectorStateFn) and self._convert_vectors:
            return CircuitStateFn.from_vector(operator.to_matrix(massive=True))
        elif isinstance(operator, ListOp) and 'Dict' in operator.get_primitives():
            return operator.traverse(self.convert)
        else:
            return operator
