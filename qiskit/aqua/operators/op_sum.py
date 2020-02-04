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

""" Eager Operator Sum Container """

import numpy as np
import copy
import itertools

from .op_combo_base import OpCombo


class OpSum(OpCombo):

    def __init__(self, oplist, coeff=1.0):
        """
        Args:
            oplist (list(OperatorBase)): The operators being summed.
        """
        super().__init__(oplist, coeff=coeff, combo_fn=sum)

    @property
    def num_qubits(self):
        return self.oplist[0].num_qubits

    # TODO change to *other to efficiently handle lists?
    def add(self, other):
        """ Addition. Overloaded by + in OperatorBase. """
        if self == other:
            return self.mul(2.0)
        elif isinstance(other, OpSum):
            return OpSum(self.ops + other.oplist)
        elif other in self.oplist:
            new_oplist = copy.copy(self.oplist)
            other_index = self.oplist.index(other)
            new_oplist[other_index] = new_oplist[other_index] + other
            return OpSum(new_oplist)
        return OpSum(self.ops + [other])

    # TODO implement override, given permutation invariance?
    # def equals(self, other):
    #     """ Evaluate Equality. Overloaded by == in OperatorBase. """
    #     if not isinstance(other, OpSum) or not len(self.oplist) == len(other.oplist):
    #         return False
    #     # TODO test this a lot
    #     # Should be sorting invariant, if not done stupidly
    #     return set(self.oplist) == set(other.oplist)

    # Maybe not necessary, given parent and clean combination function.
    # def to_matrix(self, massive=False):
    #     """ Return numpy matrix of operator, warn if more than 16 qubits to force the user to set massive=True if
    #     they want such a large matrix. Generally big methods like this should require the use of a converter,
    #     but in this case a convenience method for quick hacking and access to classical tools is appropriate. """
    #
    #     if self.num_qubits > 16 and not massive:
    #         # TODO figure out sparse matrices?
    #         raise ValueError('to_matrix will return an exponentially large matrix, in this case {0}x{0} elements.'
    #                          ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))
    #
    #     return sum([op.to_matrix() for op in self.oplist])
