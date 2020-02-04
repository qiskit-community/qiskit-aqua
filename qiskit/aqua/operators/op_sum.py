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

from . import OperatorBase, OpPrimitive, OpKron, OpComposition, OpVec


class OpSum(OperatorBase):

    def __init__(self, oplist, coeff=1.0):
        """
        Args:
            oplist (list(OperatorBase)): The operators being summed.
        """
        self._oplist = oplist
        # self._coeff = coeff

    @property
    def oplist(self):
        return self._oplist

    # @property
    # def coeff(self):
    #     return self._coeff

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

    def neg(self):
        """ Negate. Overloaded by - in OperatorBase. """
        return self.mul(-1.0)

    def adjoint(self):
        """ Return operator adjoint (conjugate transpose). Overloaded by ~ in OperatorBase. """
        return OpSum([op.adjoint() for op in self.oplist])

    def equals(self, other):
        """ Evaluate Equality. Overloaded by == in OperatorBase. """
        if not isinstance(other, OpSum) or not len(self.oplist) == len(other.oplist):
            return False
        # TODO test this a lot
        return self.oplist == other.oplist

    def mul(self, scalar):
        """ Scalar multiply. Overloaded by * in OperatorBase. """
        if not isinstance(scalar, (float, complex)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))
        return OpSum([op.mul(scalar) for op in self.oplist])

    # TODO figure out lazy options...
    def kron(self, other):
        """ Kron
        Note: You must be conscious of Qiskit's big-endian bit printing convention. Meaning, X.kron(Y)
        produces an X on qubit 0 and an Y on qubit 1, or X⨂Y, but would produce a QuantumCircuit which looks like
        -[Y]-
        -[X]-
        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.
        """
        # TODO do this lazily for some primitives (Matrix), and eager for others (Pauli, Instruction)?
        # if isinstance(other, OpPrimitive):
        #     return OpSum([op.kron(other) for op in self.oplist])
        # TODO is this a terrible idea....? Kronpower will probably explode. Can we collapse it inplace?
        # elif isinstance(other, OpSum):
        #     return OpSum([op1.kron(op2) for (op1, op2) in itertools.product(self.oplist, other.oplist)])

        return OpKron([self, other])

    def kronpower(self, other):
        """ Kron with Self Multiple Times """
        if not isinstance(other, int) or other <= 0:
            raise TypeError('Kronpower can only take positive int arguments')
        return OpKron([self]*other)

    # TODO change to *other to efficiently handle lists?
    def compose(self, other):
        """ Operator Composition (Linear algebra-style, right-to-left)

        Note: You must be conscious of Quantum Circuit vs. Linear Algebra ordering conventions. Meaning, X.compose(Y)
        produces an X∘Y on qubit 0, but would produce a QuantumCircuit which looks like
        -[Y]-[X]-
        Because Terra prints circuits with the initial state at the left side of the circuit.
        """

        # TODO do this lazily for some primitives (Matrix), and eager for others (Pauli, Instruction)?
        # if isinstance(other, OpPrimitive):
        #     return OpSum([op.compose(other) for op in self.oplist])
        # TODO is this a terrible idea....? Power will probably explode. Can we collapse it inplace?
        # elif isinstance(other, OpSum):
        #     return OpSum([op1.compose(op2) for (op1, op2) in itertools.product(self.oplist, other.oplist)])

        return OpComposition([self, other])

    def power(self, other):
        """ Compose with Self Multiple Times """
        if not isinstance(other, int) or other <= 0:
            raise TypeError('power can only take positive int arguments')
        temp = OpComposition([self]*other)

    def to_matrix(self, massive=False):
        """ Return numpy matrix of operator, warn if more than 16 qubits to force the user to set massive=True if
        they want such a large matrix. Generally big methods like this should require the use of a converter,
        but in this case a convenience method for quick hacking and access to classical tools is appropriate. """

        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError('to_matrix will return an exponentially large matrix, in this case {0}x{0} elements.'
                             ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        return sum([op.to_matrix() for op in self.oplist])

    # TODO print Instructions as drawn circuits
    def __str__(self):
        """Overload str() """
        return "Sum of {}".format([str(op) for op in self.oplist])

    def __repr__(self):
        """Overload str() """
        return "OpSum({})".format(repr(self.oplist))

    def print_details(self):
        """ print details """
        raise NotImplementedError
