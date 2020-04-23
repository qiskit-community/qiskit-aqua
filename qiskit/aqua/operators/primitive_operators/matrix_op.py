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

""" MatrixOp Class """

from typing import Union, Optional, Set
import logging
import numpy as np
from scipy.sparse import spmatrix

from qiskit.quantum_info import Operator as MatrixOperator
from qiskit.circuit import ParameterExpression, Instruction
from qiskit.extensions.hamiltonian_gate import HamiltonianGate

from ..operator_base import OperatorBase
from ..primitive_operators.circuit_op import CircuitOp
from ..combo_operators.summed_op import SummedOp
from ..combo_operators.composed_op import ComposedOp
from ..combo_operators.tensored_op import TensoredOp
from .primitive_op import PrimitiveOp

logger = logging.getLogger(__name__)


class MatrixOp(PrimitiveOp):
    """ Class for Operators represented by matrices, backed by Terra's ``Operator`` module.

    """

    def __init__(self, primitive: Union[list, np.ndarray, spmatrix, MatrixOperator] = None,
                 coeff: Optional[Union[int, float, complex, ParameterExpression]] = 1.0) -> None:
        """
        Args:
            primitive ([[complex]], np.ndarray, MatrixOperator, spmatrix): The matrix-like object
                which defines the behavior of the underlying function.
            coeff (int, float, complex, ParameterExpression): A coefficient multiplying the
                primitive

        Raises:
            TypeError: invalid parameters.
            ValueError: invalid parameters.
        """
        if isinstance(primitive, spmatrix):
            primitive = primitive.toarray()

        if isinstance(primitive, (list, np.ndarray)):
            primitive = MatrixOperator(primitive)

        if not isinstance(primitive, MatrixOperator):
            raise TypeError(
                'MatrixOp can only be instantiated with MatrixOperator, '
                'not {}'.format(type(primitive)))

        if not primitive.input_dims() == primitive.output_dims():
            raise ValueError('Cannot handle non-square matrices yet.')

        super().__init__(primitive, coeff=coeff)

    def primitive_strings(self) -> Set[str]:
        return {'Matrix'}

    @property
    def num_qubits(self) -> int:
        return len(self.primitive.input_dims())

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over operators with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        if isinstance(other, MatrixOp):
            return MatrixOp((self.coeff * self.primitive) + (other.coeff * other.primitive))

        # Covers Paulis, Circuits, and all else.
        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return MatrixOp(self.primitive.conjugate().transpose(), coeff=np.conj(self.coeff))

    def equals(self, other: OperatorBase) -> bool:
        if not isinstance(other, PrimitiveOp) \
                or not isinstance(self.primitive, type(other.primitive)) \
                or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive
        # Will return NotImplementedError if not supported

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if isinstance(other.primitive, MatrixOperator):
            return MatrixOp(self.primitive.tensor(other.primitive), coeff=self.coeff * other.coeff)

        return TensoredOp([self, other])

    def compose(self, other: OperatorBase) -> OperatorBase:
        other = self._check_zero_for_composition_and_expand(other)

        if isinstance(other, MatrixOp):
            return MatrixOp(self.primitive.compose(other.primitive, front=True),
                            coeff=self.coeff * other.coeff)

        return ComposedOp([self, other])

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_matrix will return an exponentially large matrix, '
                'in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        return self.primitive.data * self.coeff

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return prim_str
        else:
            return "{} * {}".format(self.coeff, prim_str)

    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        # For other ops' eval we return self.to_matrix_op() here, but that's unnecessary here.
        if front is None:
            return self

        # pylint: disable=cyclic-import,import-outside-toplevel
        from ..combo_operators import ListOp
        from ..state_functions import StateFn, OperatorStateFn

        new_front = None

        # For now, always do this. If it's not performant, we can be more granular.
        if not isinstance(front, OperatorBase):
            front = StateFn(front, is_measurement=False)

        if isinstance(front, ListOp) and front.distributive:
            new_front = front.combo_fn([self.eval(front.coeff * front_elem)
                                        for front_elem in front.oplist])

        elif isinstance(front, OperatorStateFn):
            new_front = OperatorStateFn(self.adjoint().compose(front.to_matrix_op()).compose(self))

        elif isinstance(front, OperatorBase):
            new_front = StateFn(self.to_matrix() @ front.to_matrix())

        return new_front

    def exp_i(self) -> OperatorBase:
        """Return a ``CircuitOp`` equivalent to e^-iH for this operator H"""
        return CircuitOp(HamiltonianGate(self.primitive, time=self.coeff))

    # Op Conversions

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        return self

    def to_instruction(self) -> Instruction:
        return PrimitiveOp(self.primitive.to_instruction(), coeff=self.coeff)
