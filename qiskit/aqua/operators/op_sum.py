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

from . import OperatorBase, OpPrimitive


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
        """ Scalar multiply. Overloaded by * in OperatorBase.

        Doesn't multiply MatrixOperator until to_matrix() is called to keep things lazy and avoid big copies.
        TODO figure out if this is a bad idea.
         """
        if not isinstance(scalar, (float, complex)):
            raise ValueError('Operators can only be scalar multiplied by float or complex.')
        return OpPrimitive(self.primitive, coeff=self.coeff * scalar)

    # TODO change to *other to handle lists? How aggressively to handle pairwise business?
    def kron(self, other):
        """ Kron
        Note: You must be conscious of Qiskit's big-endian bit printing convention. Meaning, X.kron(Y)
        produces an X on qubit 0 and an Y on qubit 1, or X⨂Y, but would produce a QuantumCircuit which looks like
        -[Y]-
        -[X]-
        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.
        """
        # TODO accept primitives directly in addition to OpPrimitive?

        self_primitive, other_primitive = self.interopt_pauli_and_gate(other)

        # Both Paulis
        if isinstance(self_primitive, Pauli) and isinstance(other_primitive, Pauli):
            # TODO change Pauli kron in Terra to have optional inplace
            op_copy = Pauli(x=other_primitive.x, z=other_primitive.z)
            return OpPrimitive(op_copy.kron(self_primitive), coeff=self.coeff * other.coeff)
            # TODO double check coeffs logic for paulis

        # Both Instructions/Circuits
        elif isinstance(self_primitive, Instruction) and isinstance(other_primitive, Instruction):
            new_qc = QuantumCircuit(self_primitive.num_qubits+other_primitive.num_qubits)
            new_qc.append(self_primitive, new_qc.qubits[0:self_primitive.num_qubits])
            new_qc.append(other_primitive, new_qc.qubits[other_primitive.num_qubits:])
            # TODO Fix because converting to dag just to append is nuts
            # TODO Figure out what to do with cbits?
            return OpPrimitive(new_qc.decompose().to_instruction(), coeff=self.coeff * other.coeff)

        # Both Matrices
        elif isinstance(self_primitive, MatrixOperator) and isinstance(other_primitive, MatrixOperator):
            return OpPrimitive(self_primitive.tensor(other_primitive), coeff=self.coeff * other.coeff)

        # User custom kron-able primitive - Identical to Pauli above for now, but maybe remove deepcopy later
        elif isinstance(self_primitive, type(other_primitive)) and hasattr(self_primitive, 'kron'):
            op_copy = copy.deepcopy(other_primitive)
            return OpPrimitive(op_copy.kron(self_primitive), coeff=self.coeff * other.coeff)

        else:
            return OpKron([self_primitive, other_primitive])

    def kronpower(self, other):
        """ Kron with Self Multiple Times """
        if not isinstance(other, int) or other <= 0:
            raise TypeError('Kronpower can only take positive int arguments')
        temp = OpPrimitive(self.primitive, coeff=self.coeff)
        for i in range(other-1):
            temp = temp.kron(self)
        return temp

    # TODO change to *other to efficiently handle lists?
    def compose(self, other):
        """ Operator Composition (Linear algebra-style, right-to-left)

        Note: You must be conscious of Quantum Circuit vs. Linear Algebra ordering conventions. Meaning, X.compose(Y)
        produces an X∘Y on qubit 0, but would produce a QuantumCircuit which looks like
        -[Y]-[X]-
        Because Terra prints circuits with the initial state at the left side of the circuit.
        """
        # TODO accept primitives directly in addition to OpPrimitive?

        if not self.num_qubits == other.num_qubits:
            raise ValueError('Composition is not defined over Operators of different dimension')

        self_primitive, other_primitive = self.interopt_pauli_and_gate(other)

        # Both Paulis
        if isinstance(self_primitive, Pauli) and isinstance(other_primitive, Pauli):
            return OpPrimitive(self_primitive * other_primitive, coeff=self.coeff * other.coeff)
            # TODO double check coeffs logic for paulis

        # Both Instructions/Circuits
        elif isinstance(self_primitive, Instruction) and isinstance(other_primitive, Instruction):
            new_qc = QuantumCircuit(self_primitive.num_qubits)
            new_qc.append(other_primitive, qargs=range(self_primitive.num_qubits))
            new_qc.append(self_primitive, qargs=range(self_primitive.num_qubits))
            # TODO Fix because converting to dag just to append is nuts
            # TODO Figure out what to do with cbits?
            return OpPrimitive(new_qc.decompose().to_instruction(), coeff=self.coeff * other.coeff)

        # Both Matrices
        elif isinstance(self_primitive, MatrixOperator) and isinstance(other_primitive, MatrixOperator):
            return OpPrimitive(self_primitive.compose(other_primitive, front=True), coeff=self.coeff * other.coeff)

        # User custom compose-able primitive
        elif isinstance(self_primitive, type(other_primitive)) and hasattr(self_primitive, 'compose'):
            op_copy = copy.deepcopy(other_primitive)
            return OpPrimitive(op_copy.compose(self_primitive), coeff=self.coeff * other.coeff)

        else:
            return OpComposition([self_primitive, other_primitive])

    def power(self, other):
        """ Compose with Self Multiple Times """
        if not isinstance(other, int) or other <= 0:
            raise TypeError('power can only take positive int arguments')
        temp = OpPrimitive(self.primitive, coeff=self.coeff)
        for i in range(other - 1):
            temp = temp.compose(self)
        return temp

    def to_matrix(self, massive=False):
        """ Return numpy matrix of operator, warn if more than 16 qubits to force the user to set massive=True if
        they want such a large matrix. Generally big methods like this should require the use of a converter,
        but in this case a convenience method for quick hacking and access to classical tools is appropriate. """

        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError('to_matrix will return an exponentially large matrix, in this case {0}x{0} elements.'
                             ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        # Pauli
        if isinstance(self.primitive, Pauli):
            return self.primitive.to_matrix() * self.coeff

        # Matrix
        elif isinstance(self.primitive, MatrixOperator):
            return self.primitive.data * self.coeff

        # Both Instructions/Circuits
        elif isinstance(self.primitive, Instruction):
            qc = QuantumCircuit(self.primitive.num_qubits)
            qc.append(self.primitive, qargs=range(self.primitive.num_qubits))
            unitary = execute(qc, BasicAer.get_backend('unitary_simulator')).result().get_unitary()
            return unitary * self.coeff

        # User custom matrix-able primitive
        elif hasattr(self.primitive, 'to_matrix'):
            return self.primitive.to_matrix() * self.coeff

        else:
            raise NotImplementedError

    # TODO print Instructions as drawn circuits
    def __str__(self):
        """Overload str() """
        return "{} * {}".format(self.coeff, str(self.primitive))

    def __repr__(self):
        """Overload str() """
        return "OpPrimitive({}, coeff={}".format(repr(self.primitive), self.coeff)

    def print_details(self):
        """ print details """
        raise NotImplementedError
