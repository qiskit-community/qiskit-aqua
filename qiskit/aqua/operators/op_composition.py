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
from functools import reduce, partial

from .op_vec import OpVec


class OpComposition(OpVec):

    def __init__(self, oplist, coeff=1.0):
        """
        Args:
            oplist (list(OperatorBase)): The operators being summed.
            coeff (int, float, complex): A coefficient multiplying the primitive
        """
        super().__init__(oplist, combo_fn=partial(reduce, np.dot), coeff=coeff)

    @property
    def num_qubits(self):
        return self.oplist[0].num_qubits

    # TODO: need to kron all others with identity so dims are right? Maybe just delete this.
    # def kron(self, other):
    #     """ Kron. We only need to Kron to the last element in the composition. """
    #     return OpComposition(self.oplist[:-1] + [self.oplist[-1].kron(other)], coeff=self.coeff)

    # TODO take advantage of the mixed product property, kronpower each element in the composition
    # def kronpower(self, other):
    #     """ Kron with Self Multiple Times """
    #     raise NotImplementedError

    def adjoint(self):
        return OpComposition([op.adjoint() for op in reversed(self.oplist)], coeff=self.coeff)

    def compose(self, other):
        """ Operator Composition (Circuit-style, left to right) """
        if isinstance(other, OpComposition):
            return OpComposition(self.oplist + other.oplist, coeff=self.coeff*other.coeff)
        return OpComposition(self.oplist + [other], coeff=self.coeff)

    def eval(self, front=None, back=None):
        """ A square binary Operator can be defined as a function over two binary strings of equal length. This
        method returns the value of that function for a given pair of binary strings. For more information,
        see the eval method in operator_base.py.


        """
        # TODO do this for real later. Requires allowing Ops to take a state and return another. Can't do this yet.
        # front_holder = front
        # # Start from last op, and stop before op 0, then eval op 0 with back
        # for op in self.oplist[-1:0:-1]:
        #     front_holder = op.eval(front=front_holder)
        # return self.oplist[0].eval(front_holder, back)

        comp_mat_op = OpPrimitive(self.combo_fn([op.to_matrix() for op in self.oplist]), coeff=self.coeff)
        return comp_mat_op.eval(front=front, back=back)
