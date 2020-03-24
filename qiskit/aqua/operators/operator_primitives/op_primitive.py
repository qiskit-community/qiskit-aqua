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

import logging
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.quantum_info import Pauli
from qiskit.quantum_info import Operator as MatrixOperator

from ..operator_base import OperatorBase

logger = logging.getLogger(__name__)


class OpPrimitive(OperatorBase):
    """ Class for Wrapping Operator Primitives

    Note that all mathematical methods are not in-place,
    meaning that they return a new object, but the underlying
    primitives are not copied.

    """

    @staticmethod
    # pylint: disable=unused-argument,inconsistent-return-statements
    def __new__(cls, primitive=None, coeff=1.0):
        """ A factory method to produce the correct type of OpPrimitive subclass
        based on the primitive passed in. Primitive and coeff arguments are passed into
        subclass's init() as-is automatically by new()."""
        if cls.__name__ != 'OpPrimitive':
            return super().__new__(cls)

        # pylint: disable=cyclic-import,import-outside-toplevel
        if isinstance(primitive, (Instruction, QuantumCircuit)):
            from .op_circuit import OpCircuit
            return OpCircuit.__new__(OpCircuit)

        if isinstance(primitive, (list, np.ndarray, MatrixOperator)):
            from .op_matrix import OpMatrix
            return OpMatrix.__new__(OpMatrix)

        if isinstance(primitive, Pauli):
            from .op_pauli import OpPauli
            return OpPauli.__new__(OpPauli)

    def __init__(self, primitive, coeff=1.0):
        """
                Args:
                    primitive (Gate, Pauli, [[complex]], np.ndarray,
                    QuantumCircuit, Instruction): The operator primitive being
                    wrapped.
                    coeff (int, float, complex, ParameterExpression): A coefficient
                    multiplying the primitive
                """
        self._primitive = primitive
        self._coeff = coeff

    @property
    def primitive(self):
        """ returns primitive """
        return self._primitive

    @property
    def coeff(self):
        """ returns coeff """
        return self._coeff

    def neg(self):
        """ Negate. Overloaded by - in OperatorBase. """
        return self.mul(-1.0)

    @property
    def num_qubits(self):
        raise NotImplementedError

    def get_primitives(self):
        raise NotImplementedError

    def add(self, other):
        raise NotImplementedError

    def adjoint(self):
        """ Return operator adjoint (conjugate transpose). Overloaded by ~ in OperatorBase. """
        raise NotImplementedError

    def equals(self, other):
        raise NotImplementedError

    def mul(self, scalar):
        """ Scalar multiply. Overloaded by * in OperatorBase.

        Doesn't multiply MatrixOperator until to_matrix()
        is called to keep things lazy and avoid big copies.
         """
        # TODO figure out if this is a bad idea.
        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))
        return self.__class__(self.primitive, coeff=self.coeff * scalar)

    def kron(self, other):
        raise NotImplementedError

    def kronpower(self, other):
        """ Kron with Self Multiple Times """
        # Hack to make Z^(I^0) work as intended.
        if other == 0:
            return 1
        if not isinstance(other, int) or other < 0:
            raise TypeError('Kronpower can only take positive int arguments')
        temp = OpPrimitive(self.primitive, coeff=self.coeff)
        for _ in range(other - 1):
            temp = temp.kron(self)
        return temp

    def compose(self, other):
        raise NotImplementedError

    def _check_zero_for_composition_and_expand(self, other):
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

    def power(self, other):
        """ Compose with Self Multiple Times """
        if not isinstance(other, int) or other <= 0:
            raise TypeError('power can only take positive int arguments')
        temp = OpPrimitive(self.primitive, coeff=self.coeff)
        for _ in range(other - 1):
            temp = temp.compose(self)
        return temp

    def exp_i(self):
        """ Raise Operator to power e ^ (i * op)"""
        # pylint: disable=cyclic-import,import-outside-toplevel
        from qiskit.aqua.operators import OpEvolution
        return OpEvolution(self)

    # def to_matrix(self, massive=False):
    #     raise NotImplementedError

    def __str__(self):
        """Overload str() """
        raise NotImplementedError

    def __repr__(self):
        """Overload str() """
        return "OpPrimitive({}, coeff={})".format(repr(self.primitive), self.coeff)

    def eval(self, front=None, back=None):
        """ Evaluate the Operator function given one or both states. """
        return NotImplementedError

    def bind_parameters(self, param_dict):
        """ bind parameters """
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                # pylint: disable=import-outside-toplevel
                from ..operator_combos.op_vec import OpVec
                return OpVec([self.bind_parameters(param_dict) for param_dict in unrolled_dict])
            coeff_param = list(self.coeff.parameters)[0]
            if coeff_param in unrolled_dict:
                # TODO what do we do about complex?
                value = unrolled_dict[coeff_param]
                param_value = float(self.coeff.bind({coeff_param: value}))
        return self.__class__(self.primitive, coeff=param_value)

    # def print_details(self):
    #     """ print details """
    #     raise NotImplementedError

    def to_matrix(self, massive=False):
        """ Return matrix representing OpPrimitive evaluated on each pair of basis states."""
        raise NotImplementedError

    # Nothing to collapse here.
    def reduce(self):
        return self

    def to_matrix_op(self, massive=False):
        """ Return a MatrixOp for this operator. """
        # pylint: disable=import-outside-toplevel
        from .op_matrix import OpMatrix
        return OpMatrix(self.to_matrix(massive=massive))
