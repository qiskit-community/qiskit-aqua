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

""" Wrapping Operator Primitives """

from typing import Optional, Union
import logging
import numpy as np
from scipy.sparse import spmatrix

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.quantum_info import Pauli
from qiskit.quantum_info import Operator as MatrixOperator

from ..operator_base import OperatorBase

logger = logging.getLogger(__name__)


class PrimitiveOp(OperatorBase):
    """ Class for Wrapping Operator Primitives

    Note that all mathematical methods are not in-place,
    meaning that they return a new object, but the underlying
    primitives are not copied.

    """

    @staticmethod
    # pylint: disable=unused-argument,inconsistent-return-statements
    def __new__(cls,
                primitive: Union[Instruction, QuantumCircuit, list,
                                 np.ndarray, spmatrix, MatrixOperator, Pauli] = None,
                coeff: Optional[Union[int, float, complex,
                                      ParameterExpression]] = 1.0) -> OperatorBase:
        """ A factory method to produce the correct type of PrimitiveOp subclass
        based on the primitive passed in. Primitive and coeff arguments are passed into
        subclass's init() as-is automatically by new()."""
        if cls.__name__ != PrimitiveOp.__name__:
            return super().__new__(cls)

        # pylint: disable=cyclic-import,import-outside-toplevel
        if isinstance(primitive, (Instruction, QuantumCircuit)):
            from .circuit_op import CircuitOp
            return CircuitOp.__new__(CircuitOp)

        if isinstance(primitive, (list, np.ndarray, spmatrix, MatrixOperator)):
            from .matrix_op import MatrixOp
            return MatrixOp.__new__(MatrixOp)

        if isinstance(primitive, Pauli):
            from .pauli_op import PauliOp
            return PauliOp.__new__(PauliOp)

    def __init__(self,
                 primitive: Union[Instruction, QuantumCircuit, list,
                                  np.ndarray, spmatrix, MatrixOperator, Pauli] = None,
                 coeff: Optional[Union[int, float, complex, ParameterExpression]] = 1.0) -> None:
        """
            Args:
                primitive (Instruction, QuantumCircuit, list, np.ndarray, spmatrix,
                 MatrixOperator, Pauli): The operator primitive being wrapped.
                coeff (int, float, complex, ParameterExpression): A coefficient multiplying
                 the primitive.
        """
        self._primitive = primitive
        self._coeff = coeff

    @property
    def primitive(self):
        """ returns primitive in inherited class """
        return self._primitive

    @property
    def coeff(self) -> Union[int, float, complex, ParameterExpression]:
        """ returns coeff """
        return self._coeff

    def neg(self) -> OperatorBase:
        """ Negate. Overloaded by - in OperatorBase. """
        return self.mul(-1.0)

    @property
    def num_qubits(self) -> int:
        raise NotImplementedError

    def get_primitives(self) -> set:
        raise NotImplementedError

    def add(self, other: OperatorBase) -> OperatorBase:
        raise NotImplementedError

    def adjoint(self) -> OperatorBase:
        """ Return operator adjoint (conjugate transpose). Overloaded by ~ in OperatorBase. """
        raise NotImplementedError

    def equals(self, other: OperatorBase) -> bool:
        raise NotImplementedError

    def mul(self, scalar: Union[int, float, complex, ParameterExpression]) -> OperatorBase:
        """ Scalar multiply. Overloaded by * in OperatorBase.

        Doesn't multiply MatrixOperator until to_matrix()
        is called to keep things lazy and avoid big copies.
         """
        # TODO figure out if this is a bad idea.
        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))
        # Need to return self.__class__ in case the object is one of the inherited OpPrimitives
        return self.__class__(self.primitive, coeff=self.coeff * scalar)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        raise NotImplementedError

    def tensorpower(self, other: int) -> Union[OperatorBase, int]:
        """ Tensor product with Self Multiple Times """
        # Hack to make Z^(I^0) work as intended.
        if other == 0:
            return 1
        if not isinstance(other, int) or other < 0:
            raise TypeError('Tensorpower can only take positive int arguments')
        temp = PrimitiveOp(self.primitive, coeff=self.coeff)
        for _ in range(other - 1):
            temp = temp.tensor(self)
        return temp

    def compose(self, other: OperatorBase) -> OperatorBase:
        raise NotImplementedError

    def _check_zero_for_composition_and_expand(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            # pylint: disable=cyclic-import,import-outside-toplevel
            from ..operator_globals import Zero
            if other == Zero:
                # Zero is special - we'll expand it to the correct qubit number.
                other = Zero.__class__('0' * self.num_qubits)
            else:
                raise ValueError(
                    'Composition is not defined over Operators of different dimensions, {} and {}, '
                    'respectively.'.format(self.num_qubits, other.num_qubits))
        return other

    def power(self, other: int) -> OperatorBase:
        """ Compose with Self Multiple Times """
        if not isinstance(other, int) or other <= 0:
            raise TypeError('power can only take positive int arguments')
        temp = PrimitiveOp(self.primitive, coeff=self.coeff)
        for _ in range(other - 1):
            temp = temp.compose(self)
        return temp

    def exp_i(self) -> OperatorBase:
        """ Raise Operator to power e ^ (i * op)"""
        # pylint: disable=cyclic-import,import-outside-toplevel
        from qiskit.aqua.operators import EvolvedOp
        return EvolvedOp(self)

    def __str__(self) -> str:
        """Overload str() """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Overload str() """
        return "PrimitiveOp({}, coeff={})".format(repr(self.primitive), self.coeff)

    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        """ Evaluate the Operator function given one or both states. """
        raise NotImplementedError

    def bind_parameters(self, param_dict: dict) -> OperatorBase:
        """ bind parameters """
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                # pylint: disable=import-outside-toplevel
                from ..combo_operators.list_op import ListOp
                return ListOp([self.bind_parameters(param_dict) for param_dict in unrolled_dict])
            coeff_param = list(self.coeff.parameters)[0]
            if coeff_param in unrolled_dict:
                # TODO what do we do about complex?
                value = unrolled_dict[coeff_param]
                param_value = float(self.coeff.bind({coeff_param: value}))
        return self.__class__(self.primitive, coeff=param_value)

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return matrix representing PrimitiveOp evaluated on each pair of basis states."""
        raise NotImplementedError

    # Nothing to collapse here.
    def reduce(self) -> OperatorBase:
        return self

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """ Return a MatrixOp for this operator. """
        # pylint: disable=import-outside-toplevel
        from .matrix_op import MatrixOp
        return MatrixOp(self.to_matrix(massive=massive))
