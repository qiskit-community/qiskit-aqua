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

import logging
import numpy as np
import itertools

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Pauli
from qiskit.quantum_info import Operator as MatrixOperator, Statevector

from .operator_base import OperatorBase
from .state_fn import StateFn

logger = logging.getLogger(__name__)


class OpPrimitive(OperatorBase):
    """ Class for Wrapping Operator Primitives

    Note that all mathematical methods are not in-place, meaning that they return a new object, but the underlying
    primitives are not copied.

    """

    @staticmethod
    def __new__(cls, primitive=None, coeff=1.0):
        if not cls.__name__ == 'OpPrimitive':
            return super().__new__(cls)
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
                    primtive (Gate, Pauli, [[complex]], np.ndarray, QuantumCircuit, Instruction): The operator primitive being
                    wrapped.
                    coeff (int, float, complex): A coefficient multiplying the primitive
                """
        self._primitive = primitive
        self._coeff = coeff

    @property
    def primitive(self):
        return self._primitive

    @property
    def coeff(self):
        return self._coeff

    def neg(self):
        """ Negate. Overloaded by - in OperatorBase. """
        return self.mul(-1.0)

    # @property
    # def num_qubits(self):
    #     raise NotImplementedError

    # def get_primitives(self):
    #     raise NotImplementedError

    # def add(self, other):
    #     raise NotImplementedError

    # def adjoint(self):
    #     """ Return operator adjoint (conjugate transpose). Overloaded by ~ in OperatorBase. """
    #     raise NotImplementedError

    # def equals(self, other):
    #     raise NotImplementedError

    def mul(self, scalar):
        """ Scalar multiply. Overloaded by * in OperatorBase.

        Doesn't multiply MatrixOperator until to_matrix() is called to keep things lazy and avoid big copies.
        TODO figure out if this is a bad idea.
         """
        if not isinstance(scalar, (int, float, complex)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))
        return self.__class__(self.primitive, coeff=self.coeff * scalar)

    # def kron(self, other):
    #     raise NotImplementedError

    def kronpower(self, other):
        """ Kron with Self Multiple Times """
        if not isinstance(other, int) or other <= 0:
            raise TypeError('Kronpower can only take positive int arguments')
        temp = OpPrimitive(self.primitive, coeff=self.coeff)
        for i in range(other-1):
            temp = temp.kron(self)
        return temp

    # def compose(self, other):
    #     raise NotImplementedError

    def _check_zero_for_composition_and_expand(self, other):
        if not self.num_qubits == other.num_qubits:
            from . import Zero
            if other == Zero:
                # Zero is special - we'll expand it to the correct qubit number.
                other = Zero.__class__('0' * self.num_qubits)
            else:
                raise ValueError('Composition is not defined over Operators of different dimensions, {} and {}, '
                                 'respectively.'.format(self.num_qubits, other.num_qubits))
        return other

    def power(self, other):
        """ Compose with Self Multiple Times """
        if not isinstance(other, int) or other <= 0:
            raise TypeError('power can only take positive int arguments')
        temp = OpPrimitive(self.primitive, coeff=self.coeff)
        for i in range(other - 1):
            temp = temp.compose(self)
        return temp

    # def to_matrix(self, massive=False):
    #     raise NotImplementedError

    # def __str__(self):
    #     """Overload str() """
    #     raise NotImplementedError

    def __repr__(self):
        """Overload str() """
        return "OpPrimitive({}, coeff={})".format(repr(self.primitive), self.coeff)

    def print_details(self):
        """ print details """
        raise NotImplementedError

    def eval(self, front=None, back=None):
        """ A square binary Operator can be defined as a function over two binary strings of equal length. This
        method returns the value of that function for a given pair of binary strings. For more information,
        see the eval method in operator_base.py.

        Notice that Pauli evals will always return 0 for Paulis with X or Y terms if val1 == val2. This is why we must
        convert to a {Z,I}^n Pauli basis to take "averaging" style expectations (e.g. PauliExpectation).
        """

        if isinstance(front, str):
            front = {str: 1}
        if isinstance(back, str):
            front = {str: 1}

        if front is None and back is None:
            return self.to_matrix()
        elif front is None:
            # Saves having to reimplement logic twice for front and back
            return self.adjoint().eval(front=back, back=None).adjoint()

        # Pauli
        if isinstance(self.primitive, Pauli):
            if isinstance(front, dict) and isinstance(back, dict):
                sum = 0
                for (str1, str2) in itertools.product(front.keys(), back.keys()):
                    bitstr1 = np.asarray(list(str1)).astype(np.bool)
                    bitstr2 = np.asarray(list(str2)).astype(np.bool)

                    # fix_endianness
                    corrected_x_bits = self.primitive.x[::-1]
                    corrected_z_bits = self.primitive.z[::-1]

                    x_factor = np.logical_xor(bitstr1, bitstr2) == corrected_x_bits
                    z_factor = 1 - 2*np.logical_and(bitstr1, corrected_z_bits)
                    y_factor = np.sqrt(1 - 2*np.logical_and(corrected_x_bits, corrected_z_bits) + 0j)
                    sum += self.coeff * np.product(x_factor * z_factor * y_factor) * front[bitstr1] * back[bitstr1]
                return sum
            elif front and back:
                return self.eval(back).adjoint().eval(front)
            # From here on, assume back is None
            if isinstance(front, StateFn):
                if front.is_measurement:
                    raise ValueError('Operator composed with a measurement is undefined.')
                elif isinstance(front.primitive, Statevector):
                    return self.eval(front.to_matrix()) * front.coeff
                elif isinstance(front.primitive, dict):
                    return self.eval(front.primitive) * front.coeff
                elif isinstance(front.primitive, OperatorBase):
                    return self.eval(front)

            if isinstance(front, dict):
                new_dict = {}
                corrected_x_bits = self.primitive.x[::-1]
                corrected_z_bits = self.primitive.z[::-1]

                for bstr, v in front.items():
                    bitstr = np.asarray(list(bstr)).astype(np.bool)
                    new_str = np.logical_xor(bitstr, corrected_x_bits)
                    z_factor = np.product(1 - 2*np.logical_and(bitstr, corrected_z_bits))
                    y_factor = np.product(np.sqrt(1 - 2 * np.logical_and(corrected_x_bits, corrected_z_bits) + 0j))
                    new_dict[new_str] += (v*z_factor*y_factor) + new_dict.get(new_str, 0)
                return StateFn(new_dict, coeff=self.coeff)


        # Matrix
        elif isinstance(self.primitive, MatrixOperator):
            if isinstance(front, dict):
                index1 = int(front, 2)
                index2 = int(back, 2)
                return self.primitive.data[index2, index1] * self.coeff
            if isinstance(back, dict):
                pass

        # User custom eval
        elif hasattr(self.primitive, 'eval'):
            return self.primitive.eval(front, back) * self.coeff

        # Both Instructions/Circuits
        elif isinstance(self.primitive, Instruction) or hasattr(self.primitive, 'to_matrix'):
            mat = self.to_matrix()
            index1 = None if not front else int(front, 2)
            index2 = None if not front else int(back, 2)
            # Don't multiply by coeff because to_matrix() already does
            return mat[index2, index1]

        else:
            raise NotImplementedError

    # Nothing to collapse here.
    def reduce(self):
        return self
