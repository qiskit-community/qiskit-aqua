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

""" Wrapping Pauli Primitive """

import logging
import numpy as np

from qiskit.quantum_info import Operator as MatrixOperator

from ..operator_base import OperatorBase
from ..operator_combos import OpSum, OpComposition, OpKron
from .op_primitive import OpPrimitive

logger = logging.getLogger(__name__)


class OpMatrix(OpPrimitive):
    """ Class for Wrapping Pauli Primitives

    Note that all mathematical methods are not in-place, meaning that
    they return a new object, but the underlying primitives are not copied.

    """

    def __init__(self, primitive, coeff=1.0):
        """
        Args:
            primitive (Gate, Pauli, [[complex]], np.ndarray, QuantumCircuit, Instruction):
            The operator primitive being wrapped.
            coeff (int, float, complex): A coefficient multiplying the primitive
        Raises:
            TypeError: invalid parameters.
            ValueError: invalid parameters.
        """
        if isinstance(primitive, (list, np.ndarray)):
            primitive = MatrixOperator(primitive)

        if not isinstance(primitive, MatrixOperator):
            raise TypeError(
                'OpMatrix can only be instantiated with MatrixOperator, '
                'not {}'.format(type(primitive)))

        if not primitive.input_dims() == primitive.output_dims():
            raise ValueError('Cannot handle non-square matrices yet.')

        super().__init__(primitive, coeff=coeff)

    def get_primitives(self):
        """ Return a set of strings describing the primitives contained in the Operator """
        return {'Matrix'}

    # TODO replace with proper alphabets later?
    @property
    def num_qubits(self):
        return len(self.primitive.input_dims())

    # TODO change to *other to efficiently handle lists?
    def add(self, other):
        """ Addition. Overloaded by + in OperatorBase. """
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over operators with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        if isinstance(other, OpMatrix):
            return OpMatrix((self.coeff * self.primitive) + (other.coeff * other.primitive))

        # Covers Paulis, Circuits, and all else.
        return OpSum([self, other])

    def adjoint(self):
        """ Return operator adjoint (conjugate transpose). Overloaded by ~ in OperatorBase. """
        return OpMatrix(self.primitive.conjugate().transpose(), coeff=np.conj(self.coeff))

    def equals(self, other):
        """ Evaluate Equality. Overloaded by == in OperatorBase. """
        if not isinstance(other, OpPrimitive) \
                or not isinstance(self.primitive, type(other.primitive)) \
                or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive
        # Will return NotImplementedError if not supported

    # TODO change to *other to handle lists? How aggressively to handle pairwise business?
    def kron(self, other):
        """ Kron
        Note: You must be conscious of Qiskit's big-endian bit
        printing convention. Meaning, X.kron(Y)
        produces an X on qubit 0 and an Y on qubit 1, or X⨂Y,
        but would produce a QuantumCircuit which looks like
        -[Y]-
        -[X]-
        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.
        """
        if isinstance(other.primitive, MatrixOperator):
            return OpMatrix(self.primitive.tensor(other.primitive), coeff=self.coeff * other.coeff)

        return OpKron([self, other])

    # TODO change to *other to efficiently handle lists?
    def compose(self, other):
        """ Operator Composition (Linear algebra-style, right-to-left)

        Note: You must be conscious of Quantum Circuit vs. Linear Algebra ordering
        conventions. Meaning, X.compose(Y)
        produces an X∘Y on qubit 0, but would produce a QuantumCircuit which looks like
        -[Y]-[X]-
        Because Terra prints circuits with the initial state at the left side of the circuit.
        """
        # TODO accept primitives directly in addition to OpPrimitive?

        other = self._check_zero_for_composition_and_expand(other)

        if isinstance(other, OpMatrix):
            return OpMatrix(self.primitive.compose(other.primitive, front=True),
                            coeff=self.coeff * other.coeff)

        return OpComposition([self, other])

    def to_matrix(self, massive=False):
        """ Return numpy matrix of operator, warn if more than 16 qubits to force
        the user to set massive=True if
        they want such a large matrix. Generally big methods like this should require
        the use of a converter,
        but in this case a convenience method for quick hacking and access to classical
        tools is appropriate. """

        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_matrix will return an exponentially large matrix, '
                'in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        return self.primitive.data * self.coeff

    def to_matrix_op(self, massive=False):
        return self

    def __str__(self):
        """Overload str() """
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return prim_str
        else:
            return "{} * {}".format(self.coeff, prim_str)

    def eval(self, front=None):
        """ A square binary Operator can be defined as a function over two binary
        strings of equal length. This
        method returns the value of that function for a given pair of binary strings.
        For more information,
        see the eval method in operator_base.py.

        Notice that Pauli evals will always return 0 for Paulis with X or Y terms
        if val1 == val2. This is why we must
        convert to a {Z,I}^n Pauli basis to take "averaging"
        style expectations (e.g. PauliExpectation).
        """
        # For other ops eval we return self.to_matrix_op() here, but that's unnecessary here.
        if front is None:
            return self

        # pylint: disable=cyclic-import,import-outside-toplevel
        from ..operator_combos import OpVec
        from ..state_functions import StateFn, StateFnOperator

        new_front = None

        # For now, always do this. If it's not performant, we can be more granular.
        if not isinstance(front, OperatorBase):
            front = StateFn(front, is_measurement=False)

        if isinstance(front, OpVec) and front.distributive:
            new_front = front.combo_fn([self.eval(front.coeff * front_elem)
                                        for front_elem in front.oplist])

        elif isinstance(front, StateFnOperator):
            new_front = StateFnOperator(self.adjoint().compose(front.to_matrix_op()).compose(self))

        elif isinstance(front, OperatorBase):
            new_front = StateFn(self.to_matrix() @ front.to_matrix())

        return new_front

    def to_simulation_instruction(self):
        """ returns simulation instruction """
        return OpPrimitive(self.primitive.to_instruction(), coeff=self.coeff)
