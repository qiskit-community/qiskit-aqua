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
from functools import partial, reduce

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import StateFnDict, StateFnVector, StateFnCircuit, OpVec
from .converter_base import ConverterBase

logger = logging.getLogger(__name__)


class DicttoCircuitSum(ConverterBase):
    """ Very naively convert StateFnDicts to sums of StateFnCircuits which each
    prepare the bit strings in the keys of the dict."""

    def __init__(self, traverse=True, convert_dicts=True, convert_vectors=True):
        self._traverse = traverse
        self._convert_dicts = convert_dicts
        self._convert_vectors = convert_vectors

    def convert(self, operator):

        if isinstance(operator, StateFnDict) and self._convert_dicts:
            return StateFnCircuit.from_dict(operator.primitive)
        if isinstance(operator, StateFnVector) and self._convert_vectors:
            return StateFnCircuit.from_vector(operator.to_matrix(massive=True))
        elif isinstance(operator, OpVec) and 'Dict' in operator.get_primitives():
            return operator.traverse(self.convert)
        else:
            return operator
