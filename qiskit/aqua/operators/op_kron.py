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

""" Eager Operator Kron Container """

from .operator_base import OperatorBase


class OpKron(OperatorBase):

    def __init__(self, ops):
        pass

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

    def power(self, other):
        """ Compose with Self Multiple Times """
        raise NotImplementedError

    def __str__(self):
        """Overload str() """
        raise NotImplementedError

    def print_details(self):
        """ print details """
        raise NotImplementedError
