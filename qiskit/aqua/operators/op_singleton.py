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

""" Weighted Pauli Operator """

import logging
import numpy as np

from qiskit import QuantumCircuit

from .operator_base import OperatorBase

logger = logging.getLogger(__name__)


class OpSingleton(OperatorBase):
    """ Class for Convenience Operator Singletons """

    def __init__(self, primitive, name=None):
        """
        Args:
            primtive (Gate, Pauli, [[complex]], np.ndarray, QuantumCircuit): The operator primitive being wrapped.
            name (str, optional): the name of operator.
        """
        self._primitive = primitive
        self._name = name

    def add(self, other, inplace=True):
        """ Addition """
        raise NotImplementedError

    def neg(self):
        """ Negate """
        raise NotImplementedError

    def equals(self, other):
        """ Evaluate Equality """
        raise NotImplementedError

    def mul(self, scalar):
        """ Scalar multiply """
        raise NotImplementedError

    def kron(self, other):
        """ Kron """
        raise NotImplementedError

    def kronpower(self, other):
        """ Kron with Self Multiple Times """
        raise NotImplementedError

    def compose(self, other):
        """ Operator Composition (Circuit-style, left to right) """
        raise NotImplementedError

    def dot(self, other):
        """ Operator Composition (Linear algebra-style, right to left) """
        raise NotImplementedError

    def power(self, other):
        """ Compose with Self Multiple Times """
        raise NotImplementedError

    def __str__(self):
        """Overload str() """
        raise NotImplementedError

    def print_details(self):
        """ print details """
        raise NotImplementedError
