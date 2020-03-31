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

""" Eager Operator Sum Container """

from typing import List, Union
import copy
from functools import reduce, partial

from ..operator_base import OperatorBase
from .list_op import ListOp


class SummedOp(ListOp):
    """ Eager Operator Sum Container """
    def __init__(self,
                 oplist: List[OperatorBase],
                 coeff: Union[int, float, complex] = 1.0,
                 abelian: bool = False) -> None:
        """
        Args:
            oplist: The operators being summed.
            coeff: A coefficient multiplying the operator
            abelian: indicates if abelian
        """
        super().__init__(oplist, combo_fn=partial(reduce, lambda x, y: x + y),
                         coeff=coeff, abelian=abelian)

    @property
    def num_qubits(self) -> int:
        return self.oplist[0].num_qubits

    # TODO: Keep this property for evals or just enact distribution at composition time?
    @property
    def distributive(self) -> bool:
        """ Indicates whether the ListOp or subclass is distributive
        under composition. ListOp and SummedOp are,
        meaning that opv @ op = opv[0] @ op + opv[1] @
        op +... (plus for SummedOp, vec for ListOp, etc.),
        while ComposedOp and TensoredOp do not behave this way."""
        return True

    # TODO change to *other to efficiently handle lists?
    def add(self, other: OperatorBase) -> OperatorBase:
        """ Addition. Overloaded by + in OperatorBase. """
        if self == other:
            return self.mul(2.0)
        elif isinstance(other, SummedOp):
            self_new_ops = [op.mul(self.coeff) for op in self.oplist]
            other_new_ops = [op.mul(other.coeff) for op in other.oplist]
            return SummedOp(self_new_ops + other_new_ops)
        elif other in self.oplist:
            new_oplist = copy.copy(self.oplist)
            other_index = self.oplist.index(other)
            new_oplist[other_index] = new_oplist[other_index] + other
            return SummedOp(new_oplist, coeff=self.coeff)
        return SummedOp(self.oplist + [other], coeff=self.coeff)

    # TODO implement override, given permutation invariance?
    # def equals(self, other):
    #     """ Evaluate Equality. Overloaded by == in OperatorBase. """
    #     if not isinstance(other, SummedOp) or not len(self.oplist) == len(other.oplist):
    #         return False
    #     # TODO test this a lot
    #     # Should be sorting invariant, if not done stupidly
    #     return set(self.oplist) == set(other.oplist)

    # Try collapsing list or trees of Sums.
    # TODO be smarter about the fact that any two ops in oplist could be evaluated for sum.
    def reduce(self) -> OperatorBase:
        reduced_ops = [op.reduce() for op in self.oplist]
        reduced_ops = reduce(lambda x, y: x.add(y), reduced_ops) * self.coeff
        if isinstance(reduced_ops, SummedOp) and len(reduced_ops.oplist) == 1:
            return reduced_ops.oplist[0]
        else:
            return reduced_ops
