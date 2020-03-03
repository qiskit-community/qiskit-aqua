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
from qiskit.quantum_info import Statevector

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

    # TODO: Keep this property for evals or just enact distribution at composition time?
    @property
    def distributive(self):
        """ Indicates whether the OpVec or subclass is distrubtive under composition. OpVec and OpSum are,
        meaning that opv @ op = opv[0] @ op + opv[1] @ op +... (plus for OpSum, vec for OpVec, etc.),
        while OpComposition and OpKron do not behave this way."""
        return False

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
        # front_holder = front.eval(front=front)
        # Start from last op, and stop before op 0, then eval op 0 with back
        # for op in self.oplist[-1:0:-1]:
        #     front_holder = op.eval(front=front_holder)
        # return self.oplist[0].eval(front=front_holder, back)

        def tree_recursive_eval(r, l):
            # if isinstance(l, list):
            #     return [tree_recursive_eval(l_op, r) for l_op in l]
            if isinstance(r, list):
                return [tree_recursive_eval(r_op, l) for r_op in r]
            else:
                return l.eval(r)

        eval_list = self.oplist
        # Only one op needs to be multiplied, so just multiply the first.
        eval_list[0] = eval_list[0] * self.coeff
        eval_list = eval_list + [front] if front else eval_list
        if isinstance(back, (str, dict, Statevector)):
            from . import StateFn
            back = StateFn(back)
        eval_list = [back] + eval_list if back else eval_list

        return reduce(tree_recursive_eval, reversed(eval_list))

        # def tree_eval(t):
        #     if isinstance(t, list):
        #         return [tree_eval(t_op) for t_op in t]
        #     else:
        #         if len(t.shape) == 2 and t.shape[0] == t.shape[1]:
        #             from . import OpPrimitive
        #             t_mat_op = OpPrimitive(t, coeff=coeff)
        #             return t_mat_op.eval(front=front, back=back)
        #         elif t.shape == (1,):
        #             return t[0]
        #         else:
        #             from . import StateFn
        #             meas = not len(t.shape) == 1
        #             comp_mat = StateFn(t, coeff=coeff, is_measurement=meas)
        #             return comp_mat.eval(other=front)
        # return tree_eval(mat_composition_tree)

        # comp_mat_or_vec = self.combo_fn([op.to_matrix() for op in self.oplist])
        # if len(comp_mat_or_vec.shape) == 2 and comp_mat_or_vec.shape[0] == comp_mat_or_vec.shape[1]:
        #     from . import OpPrimitive
        #     comp_mat = OpPrimitive(comp_mat_or_vec, coeff=self.coeff)
        #     return comp_mat.eval(front=front, back=back)
        # elif comp_mat_or_vec.shape == (1,):
        #     return comp_mat_or_vec[0]
        # else:
        #     from . import StateFn
        #     meas = not len(comp_mat_or_vec.shape) == 1
        #     comp_mat = StateFn(comp_mat_or_vec, coeff=self.coeff, is_measurement=meas)
        #     return comp_mat.eval(other=front)

    # Try collapsing list or trees of compositions into a single <Measurement | Op | State>.
    def distribute_reduce(self):
        reduced_ops = [op.reduce() for op in self.oplist]
        reduced_ops = reduce(lambda x, y: x.compose(y), reduced_ops) * self.coeff
        if isinstance(reduced_ops, OpComposition) and len(reduced_ops.oplist) > 1:
            return reduced_ops
        else:
            return reduced_ops[0]

    def reduce(self):
        reduced_ops = [op.reduce() for op in self.oplist]

        def distribute_compose(l, r):
            if isinstance(l, OpVec) and l.distributive:
                return OpVec([distribute_compose(l_op, r) for l_op in l.oplist])
            elif isinstance(r, OpVec) and r.distributive:
                return OpVec([distribute_compose(l, r_op) for r_op in r.oplist])
            else:
                return l.compose(r)
        reduced_ops = reduce(lambda x, y: distribute_compose(x, y), reduced_ops) * self.coeff
        if isinstance(reduced_ops, OpVec) and len(reduced_ops.oplist) == 1:
            return reduced_ops.oplist[0]
        else:
            return reduced_ops
