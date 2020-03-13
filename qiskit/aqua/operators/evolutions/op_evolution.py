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

import logging
import numpy as np
import scipy

from qiskit.circuit import ParameterExpression

from ..operator_primitives import OpPrimitive
from ..operator_combos import OpSum, OpComposition, OpKron

logger = logging.getLogger(__name__)


class OpEvolution(OpPrimitive):
    """ Class for wrapping Operator Evolutions for compilation by an Evolution
    method later, essentially acting as a
    placeholder. Note that OpEvolution is a weird case of OpPrimitive.
    It happens to be that it fits into the
    OpPrimitive interface nearly perfectly, and it essentially
    represents a placeholder for an OpPrimitive later,
    even though it doesn't actually hold a primitive object. We could
    have chosen for it to be an OperatorBase,
    but would have ended up copying and pasting a lot of code from OpPrimitive."""

    def __init__(self, primitive, coeff=1.0):
        """
                Args:
                    primitive (OperatorBase): The operator being wrapped.
                    coeff (int, float, complex): A coefficient multiplying the primitive
                """
        super().__init__(primitive, coeff=coeff)

    def get_primitives(self):
        return self.primitive.get_primitives()

    @property
    def num_qubits(self):
        return self.primitive.num_qubits

    def add(self, other):
        """ Addition. Overloaded by + in OperatorBase. """
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over operators with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        if isinstance(other, OpEvolution) and self.primitive == other.primitive:
            return OpEvolution(self.primitive, coeff=self.coeff + other.coeff)

        return OpSum([self, other])

    def adjoint(self):
        """ Return operator adjoint (conjugate transpose). Overloaded by ~ in OperatorBase. """
        return OpEvolution(self.primitive.adjoint() * -1, coeff=np.conj(self.coeff))

    def equals(self, other):
        """ Evaluate Equality. Overloaded by == in OperatorBase. """
        if not isinstance(other, OpEvolution) or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive

    def kron(self, other):
        """ Kron
        Note: You must be conscious of Qiskit's big-endian bit printing
        convention. Meaning, X.kron(Y)
        produces an X on qubit 0 and an Y on qubit 1, or X⨂Y, but would produce
        a QuantumCircuit which looks
        like
        -[Y]-
        -[X]-
        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.
        """
        return OpKron([self, other])

    def compose(self, other):
        """ Operator Composition (Linear algebra-style, right-to-left)

                Note: You must be conscious of Quantum Circuit vs. Linear Algebra
                ordering conventions. Meaning,
                X.compose(Y)
                produces an X∘Y on qubit 0, but would produce a QuantumCircuit which looks like
                -[Y]-[X]-
                Because Terra prints circuits with the initial state at the
                left side of the circuit.
                """
        # TODO accept primitives directly in addition to OpPrimitive?

        other = self._check_zero_for_composition_and_expand(other)

        return OpComposition([self, other])

    def to_matrix(self, massive=False):
        prim_mat = 1.j * self.primitive.to_matrix()
        return scipy.linalg.expm(prim_mat) * self.coeff

    def __str__(self):
        """Overload str() """
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return 'e^(i*{})'.format(prim_str)
        else:
            return "{} * e^(i*{})".format(self.coeff, prim_str)

    def __repr__(self):
        """Overload str() """
        return "OpEvolution({}, coeff={})".format(repr(self.primitive), self.coeff)

    def reduce(self):
        return OpEvolution(self.primitive.reduce(), coeff=self.coeff)

    def bind_parameters(self, param_dict):
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            coeff_param = list(self.coeff.parameters)[0]
            if coeff_param in unrolled_dict:
                # TODO what do we do about complex?
                value = unrolled_dict[coeff_param]
                param_value = float(self.coeff.bind({coeff_param: value}))
        return self.__class__(self.primitive.bind_parameters(param_dict), coeff=param_value)

    def eval(self, front=None, back=None):
        """ A square binary Operator can be defined as a function over
        two binary strings of equal length. This
        method returns the value of that function for a given pair
        of binary strings. For more information,
        see the eval method in operator_base.py.

        For OpEvolutions which haven't been converted by an Evolution
        method yet, our only option is to convert to an
        OpMatrix and eval with that.
        """
        return OpPrimitive(self.to_matrix()).eval(front=front, back=back)
