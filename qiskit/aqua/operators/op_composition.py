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

""" Eager Operator Composition Container """

import numpy as np

from . import OpCombo, OpPrimitive
from functools import reduce, partial


class OpComposition(OpCombo):

    def __init__(self, oplist, coeff=1.0):
        """
        Args:
            oplist (list(OperatorBase)): The operators being summed.
        """
        super().__init__(oplist, coeff=coeff, combo_fn=np.dot)

    def kron(self, other):
        """ Kron. We only need to Kron to the last element in the composition. """
        return OpComposition(self.oplist[:-1] + [self.oplist[-1].kron(other)], coeff=self.coeff)

    # TODO take advantage of the mixed product property, kronpower each element in the composition
    # def kronpower(self, other):
    #     """ Kron with Self Multiple Times """
    #     raise NotImplementedError

    def compose(self, other):
        """ Operator Composition (Circuit-style, left to right) """
            if isinstance(other, OpComposition):
                return OpComposition(self.ops + other.oplist, coeff=self.coeff*other.coeff)
            return OpComposition(self.ops + [other], coeff=self.coeff)