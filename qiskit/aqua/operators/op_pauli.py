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
import copy

from qiskit import QuantumCircuit, BasicAer, execute
from qiskit.circuit import Instruction
from qiskit.quantum_info import Pauli
from qiskit.quantum_info import Operator as MatrixOperator

# from .operator_base import OperatorBase
from . import OpPrimitive
from . import OpSum
from . import OpComposition
from . import OpKron

logger = logging.getLogger(__name__)


class OpPauli(OpPrimitive):
    """ Class for Wrapping Pauli Primitives

    Note that all mathematical methods are not in-place, meaning that they return a new object, but the underlying
    primitives are not copied.

    """

    def __init__(self, primitive, coeff=1.0):
        """
                Args:
                    primtive (Gate, Pauli, [[complex]], np.ndarray, QuantumCircuit, Instruction): The operator primitive being
                    wrapped.
                    coeff (int, float, complex): A coefficient multiplying the primitive
                """
        super().__init__(primitive, coeff=coeff)

    @property
    def primitive(self):
        return self._primitive

    @property
    def coeff(self):
        return self._coeff

    def get_primitives(self):
        """ Return a set of strings describing the primitives contained in the Operator """
        if isinstance(self.primitive, Instruction):
            return {'Instruction'}
        elif isinstance(self.primitive, MatrixOperator):
            return {'Matrix'}
        else:
            # Includes 'Pauli'
            return {self.primitive.__class__.__name__}

    # TODO replace with proper alphabets later?
    @property
    def num_qubits(self):
        if isinstance(self.primitive, MatrixOperator):
            return len(self.primitive.input_dims())
        if isinstance(self.primitive, Pauli):
            return len(self.primitive)
        else:
            # Works for Instruction, or user custom primitive
            return self.primitive.num_qubits

    # TODO maybe change to use converter later
    def interopt_pauli_and_gate(self, other):
        """ Helper to resolve the overlap between the Terra Pauli classes and Gate classes. First checks if the
        one of the operands are a Pauli and the other is an Instruction, and if so, converts the Pauli to an
        Instruction."""

        if isinstance(self.primitive, Instruction) and isinstance(other.primitive, Pauli):
            from qiskit.aqua.operators.converters import PaulitoInstruction
            return self.primitive, PaulitoInstruction().convert_pauli(other.primitive)
        elif isinstance(self.primitive, Pauli) and isinstance(other.primitive, Instruction):
            from qiskit.aqua.operators.converters import PaulitoInstruction
            return PaulitoInstruction().convert_pauli(self.primitive), other.primitive
        return self.primitive, other.primitive

    # TODO change to *other to efficiently handle lists?
    def add(self, other):
        """ Addition. Overloaded by + in OperatorBase. """
        if not self.num_qubits == other.num_qubits:
            raise ValueError('Sum over operators with different numbers of qubits, {} and {}, is not well '
                             'defined'.format(self.num_qubits, other.num_qubits))
        if isinstance(other, OpPrimitive):
            if isinstance(self.primitive, type(other.primitive)) and self.primitive == other.primitive:
                return OpPrimitive(self.primitive, coeff=self.coeff + other.coeff)
            # Covers MatrixOperator and custom.
            elif isinstance(self.primitive, type(other.primitive)) and hasattr(self.primitive, 'add'):
                return OpPrimitive((self.coeff * self.primitive).add(other.primitive * other.coeff))

        # Covers Paulis, Circuits, and all else.
        return OpSum([self, other])

    def neg(self):
        """ Negate. Overloaded by - in OperatorBase. """
        return self.mul(-1.0)

    def adjoint(self):
        """ Return operator adjoint (conjugate transpose). Overloaded by ~ in OperatorBase. """

        # Pauli
        if isinstance(self.primitive, Pauli):
            return OpPrimitive(self.primitive, coeff=np.conj(self.coeff))

        # Matrix
        elif isinstance(self.primitive, MatrixOperator):
            return OpPrimitive(self.primitive.conjugate().transpose(), coeff=np.conj(self.coeff))

        # Both Instructions/Circuits
        elif isinstance(self.primitive, Instruction):
            return OpPrimitive(self.primitive.inverse(), coeff=np.conj(self.coeff))

        # User custom adjoint-able primitive
        elif hasattr(self.primitive, 'adjoint'):
            return OpPrimitive(self.primitive.adjoint(), coeff=np.conj(self.coeff))

        else:
            raise NotImplementedError

    def equals(self, other):
        """ Evaluate Equality. Overloaded by == in OperatorBase. """
        if not isinstance(other, OpPrimitive) \
                or not isinstance(self.primitive, type(other.primitive)) \
                or not self.coeff == other.coeff:
            return False
        return self.primitive == other.primitive
        # Will return NotImplementedError if not supported

    def mul(self, scalar):
        """ Scalar multiply. Overloaded by * in OperatorBase.

        Doesn't multiply MatrixOperator until to_matrix() is called to keep things lazy and avoid big copies.
        TODO figure out if this is a bad idea.
         """
        if not isinstance(scalar, (int, float, complex)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))
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
            # NOTE!!! REVERSING QISKIT ENDIANNESS HERE
            return OpPrimitive(op_copy.kron(self_primitive), coeff=self.coeff * other.coeff)

        # Both Instructions/Circuits
        elif isinstance(self_primitive, Instruction) and isinstance(other_primitive, Instruction):
            new_qc = QuantumCircuit(self_primitive.num_qubits+other_primitive.num_qubits)
            # NOTE!!! REVERSING QISKIT ENDIANNESS HERE
            new_qc.append(other_primitive, new_qc.qubits[0:other_primitive.num_qubits])
            new_qc.append(self_primitive, new_qc.qubits[other_primitive.num_qubits:])
            # TODO Fix because converting to dag just to append is nuts
            # TODO Figure out what to do with cbits?
            return OpPrimitive(new_qc.decompose().to_instruction(), coeff=self.coeff * other.coeff)

        # Both Matrices
        elif isinstance(self_primitive, MatrixOperator) and isinstance(other_primitive, MatrixOperator):
            return OpPrimitive(self_primitive.tensor(other_primitive), coeff=self.coeff * other.coeff)

        # User custom kron-able primitive
        elif isinstance(self_primitive, type(other_primitive)) and hasattr(self_primitive, 'kron'):
            op_copy = copy.deepcopy(other_primitive)
            return OpPrimitive(self_primitive.kron(op_copy), coeff=self.coeff * other.coeff)

        else:
            return OpKron([self, other])

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
            from . import Zero
            if other == Zero:
                # Zero is special - we'll expand it to the correct qubit number.
                from . import StateFn
                other = StateFn('0' * self.num_qubits)
            else:
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
            new_qc = new_qc.decompose()
            return OpPrimitive(new_qc.to_instruction(), coeff=self.coeff * other.coeff)

        # Both Matrices
        elif isinstance(self_primitive, MatrixOperator) and isinstance(other_primitive, MatrixOperator):
            return OpPrimitive(self_primitive.compose(other_primitive, front=True), coeff=self.coeff * other.coeff)

        # User custom compose-able primitive
        elif isinstance(self_primitive, type(other_primitive)) and hasattr(self_primitive, 'compose'):
            op_copy = copy.deepcopy(other_primitive)
            return OpPrimitive(op_copy.compose(self_primitive), coeff=self.coeff * other.coeff)

        else:
            return OpComposition([self, other])

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
            # NOTE: not reversing qubits!!
            # qc.append(self.primitive, qargs=range(self.primitive.num_qubits)[::-1])
            qc.append(self.primitive, qargs=range(self.primitive.num_qubits))
            unitary_backend = BasicAer.get_backend('unitary_simulator')
            unitary = execute(qc, unitary_backend, optimization_level=0).result().get_unitary()
            return unitary * self.coeff

        # User custom matrix-able primitive
        elif hasattr(self.primitive, 'to_matrix'):
            return self.primitive.to_matrix() * self.coeff

        else:
            raise NotImplementedError

    def __str__(self):
        """Overload str() """
        if isinstance(self.primitive, Instruction):
            qc = QuantumCircuit(self.num_qubits)
            qc.append(self.primitive, range(self.num_qubits))
            qc = qc.decompose()
            prim_str = str(qc.draw(output='text'))
            # prim_str = self.primitive.__class__.__name__
        else:
            prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return prim_str
        else:
            return "{} * {}".format(self.coeff, prim_str)

    def __repr__(self):
        """Overload str() """
        return "OpPrimitive({}, coeff={}".format(repr(self.primitive), self.coeff)

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

        # Pauli
        if isinstance(self.primitive, Pauli):
            bitstr1 = np.asarray(list(front)).astype(np.bool)
            bitstr2 = np.asarray(list(back)).astype(np.bool)

            # fix_endianness
            corrected_x_bits = self.primitive.x[::-1]
            corrected_z_bits = self.primitive.z[::-1]

            x_factor = np.logical_xor(bitstr1, bitstr2) == corrected_x_bits
            z_factor = 1 - 2*np.logical_and(bitstr1, corrected_z_bits)
            y_factor = np.sqrt(1 - 2*np.logical_and(corrected_x_bits, corrected_z_bits) + 0j)
            return self.coeff * np.product(x_factor*z_factor*y_factor)

        # Matrix
        elif isinstance(self.primitive, MatrixOperator):
            index1 = int(front, 2)
            index2 = int(back, 2)
            return self.primitive.data[index2, index1] * self.coeff

        # User custom eval
        elif hasattr(self.primitive, 'eval'):
            return self.primitive.eval(front, back) * self.coeff

        # Both Instructions/Circuits
        elif isinstance(self.primitive, Instruction) or hasattr(self.primitive, 'to_matrix'):
            mat = self.to_matrix()
            index1 = int(front, 2)
            index2 = int(back, 2)
            # Don't multiply by coeff because to_matrix() already does
            return mat[index2, index1]

        else:
            raise NotImplementedError

    # Nothing to collapse here.
    def reduce(self):
        return self
