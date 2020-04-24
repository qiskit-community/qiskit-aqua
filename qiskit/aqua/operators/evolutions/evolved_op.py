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

""" Wrapping Operator Evolutions """

from typing import Optional, Union, Set
import logging
import numpy as np
import scipy

from qiskit.circuit import ParameterExpression, Instruction

from ..operator_base import OperatorBase
from ..primitive_ops.primitive_op import PrimitiveOp
from ..list_ops.summed_op import SummedOp
from ..list_ops.composed_op import ComposedOp
from ..list_ops.tensored_op import TensoredOp

logger = logging.getLogger(__name__)


class EvolvedOp(PrimitiveOp):
    """ Class for wrapping Operator Evolutions for compilation by an Evolution
    method later, essentially acting as a
    placeholder. Note that EvolvedOp is a weird case of PrimitiveOp.
    It happens to be that it fits into the
    PrimitiveOp interface nearly perfectly, and it essentially
    represents a placeholder for an PrimitiveOp later,
    even though it doesn't actually hold a primitive object. We could
    have chosen for it to be an OperatorBase,
    but would have ended up copying and pasting a lot of code from PrimitiveOp."""

    def __init__(self,
                 primitive: OperatorBase,
                 coeff: Optional[Union[int, float, complex, ParameterExpression]] = 1.0) -> None:
        """
        Args:
            primitive: The operator being wrapped.
            coeff: A coefficient multiplying the operator
        """
        super().__init__(primitive, coeff=coeff)

    def primitive_strings(self) -> Set[str]:
        return self.primitive.primitive_strings()

    @property
    def num_qubits(self) -> int:
        return self.primitive.num_qubits

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over operators with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        if isinstance(other, EvolvedOp) and self.primitive == other.primitive:
            return EvolvedOp(self.primitive, coeff=self.coeff + other.coeff)

        if isinstance(other, SummedOp):
            return SummedOp([self] + other.oplist)

        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return EvolvedOp(self.primitive.adjoint() * -1, coeff=np.conj(self.coeff))

    def equals(self, other: OperatorBase) -> bool:
        if not isinstance(other, EvolvedOp) or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if isinstance(other, TensoredOp):
            return TensoredOp([self] + other.oplist)

        return TensoredOp([self, other])

    def compose(self, other: OperatorBase) -> OperatorBase:
        """ Operator Composition (Linear algebra-style, right-to-left)

        Note: You must be conscious of Quantum Circuit vs. Linear Algebra
        ordering conventions. Meaning,
        X.compose(Y)
        produces an X∘Y on qubit 0, but would produce a QuantumCircuit which looks like

            -[Y]-[X]-

        Because Terra prints circuits with the initial state at the
        left side of the circuit.
        """
        other = self._check_zero_for_composition_and_expand(other)

        if isinstance(other, ComposedOp):
            return ComposedOp([self] + other.oplist)

        return ComposedOp([self, other])

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return 'e^(-i*{})'.format(prim_str)
        else:
            return "{} * e^(-i*{})".format(self.coeff, prim_str)

    def __repr__(self) -> str:
        """Overload str() """
        return "EvolvedOp({}, coeff={})".format(repr(self.primitive), self.coeff)

    def reduce(self) -> OperatorBase:
        return EvolvedOp(self.primitive.reduce(), coeff=self.coeff)

    def bind_parameters(self, param_dict: dict) -> OperatorBase:
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                # pylint: disable=import-outside-toplevel
                from ..list_ops.list_op import ListOp
                return ListOp([self.bind_parameters(param_dict) for param_dict in unrolled_dict])
            coeff_param = list(self.coeff.parameters)[0]
            if coeff_param in unrolled_dict:
                # TODO what do we do about complex?
                value = unrolled_dict[coeff_param]
                param_value = float(self.coeff.bind({coeff_param: value}))
        return EvolvedOp(self.primitive.bind_parameters(param_dict), coeff=param_value)

    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        """ A square binary Operator can be defined as a function over
        two binary strings of equal length. This
        method returns the value of that function for a given pair
        of binary strings. For more information,
        see the eval method in operator_base.py.

        For EvolvedOps which haven't been converted by an Evolution
        method yet, our only option is to convert to an
        MatrixOp and eval with that.
        """
        return self.to_matrix_op().eval(front=front)

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        """ returns matrix """
        prim_mat = -1.j * self.primitive.to_matrix()
        # pylint: disable=no-member
        return scipy.linalg.expm(prim_mat) * self.coeff

    # pylint: disable=arguments-differ
    def to_instruction(self, massive: bool = False) -> Instruction:
        """ Return an Instruction for this operator. """
        return self.primitive.to_matrix_op(massive=massive).exp_i()
