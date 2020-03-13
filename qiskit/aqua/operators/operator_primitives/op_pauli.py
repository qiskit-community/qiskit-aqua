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
import itertools
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.extensions.standard import RZGate, RYGate, RXGate

from . import OpPrimitive
from ..operator_combos import OpSum, OpComposition, OpKron

logger = logging.getLogger(__name__)


class OpPauli(OpPrimitive):
    """ Class for Wrapping Pauli Primitives

    Note that all mathematical methods are not in-place,
    meaning that they return a new object, but the underlying
    primitives are not copied.

    """

    def __init__(self, primitive, coeff=1.0):
        """
        Args:
            primitive (Gate, Pauli, [[complex]], np.ndarray, QuantumCircuit, Instruction):
            The operator primitive being wrapped.
            coeff (int, float, complex): A coefficient multiplying the primitive
        """
        if not isinstance(primitive, Pauli):
            raise TypeError(
                'OpPauli can only be instantiated with Pualis, not {}'.format(type(primitive)))
        super().__init__(primitive, coeff=coeff)

    def get_primitives(self):
        """ Return a set of strings describing the primitives contained in the Operator """
        return {'Pauli'}

    # TODO replace with proper alphabets later?
    @property
    def num_qubits(self):
        return len(self.primitive)

    # TODO change to *other to efficiently handle lists?
    def add(self, other):
        """ Addition. Overloaded by + in OperatorBase. """
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over operators with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        if isinstance(other, OpPauli) and self.primitive == other.primitive:
            return OpPauli(self.primitive, coeff=self.coeff + other.coeff)

        return OpSum([self, other])

    def adjoint(self):
        """ Return operator adjoint (conjugate transpose). Overloaded by ~ in OperatorBase. """
        return OpPauli(self.primitive, coeff=np.conj(self.coeff))

    def equals(self, other):
        """ Evaluate Equality. Overloaded by == in OperatorBase. """
        if not isinstance(other, OpPauli) or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive

    # TODO change to *other to handle lists? How aggressively to handle pairwise business?
    def kron(self, other):
        """ Kron
        Note: You must be conscious of Qiskit's big-endian bit
        printing convention. Meaning, X.kron(Y)
        produces an X on qubit 0 and an Y on qubit 1, or X⨂Y,
        but would produce a
        QuantumCircuit which looks like
        -[Y]-
        -[X]-
        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.
        """
        # TODO accept primitives directly in addition to OpPrimitive?

        # Both Paulis
        if isinstance(other, OpPauli):
            # TODO change Pauli kron in Terra to have optional in place
            op_copy = Pauli(x=other.primitive.x, z=other.primitive.z)
            # NOTE!!! REVERSING QISKIT ENDIANNESS HERE
            return OpPauli(op_copy.kron(self.primitive), coeff=self.coeff * other.coeff)

        # Both Instructions/Circuits
        # pylint: disable=cyclic-import,import-outside-toplevel
        from . import OpCircuit
        if isinstance(other, OpCircuit):
            from qiskit.aqua.operators.converters import PaulitoInstruction
            converted_primitive = PaulitoInstruction().convert_pauli(self.primitive)
            new_qc = QuantumCircuit(self.num_qubits + other.num_qubits)
            # NOTE!!! REVERSING QISKIT ENDIANNESS HERE
            new_qc.append(other.primitive, new_qc.qubits[0:other.num_qubits])
            new_qc.append(converted_primitive, new_qc.qubits[other.num_qubits:])
            # TODO Fix because converting to dag just to append is nuts
            # TODO Figure out what to do with cbits?
            return OpCircuit(new_qc.decompose().to_instruction(), coeff=self.coeff * other.coeff)

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

        # If self is identity, just return other.
        if not any(self.primitive.x + self.primitive.z):
            return (other * self.coeff)

        # Both Paulis
        if isinstance(other, OpPauli):
            product, phase = Pauli.sgn_prod(self.primitive, other.primitive)
            return OpPrimitive(product, coeff=self.coeff * other.coeff * phase)

        # pylint: disable=cyclic-import,import-outside-toplevel
        from . import OpCircuit
        from .. import StateFnCircuit
        if isinstance(other, (OpCircuit, StateFnCircuit)):
            from qiskit.aqua.operators.converters import PaulitoInstruction
            converted_primitive = PaulitoInstruction().convert_pauli(self.primitive)
            new_qc = QuantumCircuit(self.num_qubits)
            new_qc.append(other.primitive, qargs=range(self.num_qubits))
            new_qc.append(converted_primitive, qargs=range(self.num_qubits))
            # TODO Fix because converting to dag just to append is nuts
            # TODO Figure out what to do with cbits?
            if isinstance(other, StateFnCircuit):
                return StateFnCircuit(new_qc.decompose().to_instruction(),
                                      is_measurement=other.is_measurement,
                                      coeff=self.coeff * other.coeff)
            else:
                return OpCircuit(new_qc.decompose().to_instruction(),
                                 coeff=self.coeff * other.coeff)

        return OpComposition([self, other])

    def to_matrix(self, massive=False):
        """ Return numpy matrix of operator, warn if more than
        16 qubits to force the user to set massive=True if
        they want such a large matrix. Generally big methods like
        this should require the use of a converter,
        but in this case a convenience method for quick hacking
        and access to classical tools is appropriate. """

        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError(
                'to_matrix will return an exponentially large matrix, '
                'in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        return self.primitive.to_matrix() * self.coeff

    def __str__(self):
        """Overload str() """
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return prim_str
        else:
            return "{} * {}".format(self.coeff, prim_str)

    def eval(self, front=None, back=None):
        """ A square binary Operator can be defined as a function over
        two binary strings of equal length. This
        method returns the value of that function for a given pair of
        binary strings. For more information,
        see the eval method in operator_base.py.

        Notice that Pauli evals will always return 0 for Paulis with X or Y terms
        if val1 == val2. This is why we must
        convert to a {Z,I}^n Pauli basis to take "averaging"
        style expectations (e.g. PauliExpectation).
        """

        if front is None and back is None:
            return self.to_matrix()
        elif front is None:
            # Saves having to reimplement logic twice for front and back
            return self.adjoint().eval(front=back).adjoint()
        # pylint: disable=import-outside-toplevel
        from .. import OperatorBase, StateFn, StateFnDict, StateFnVector, StateFnOperator, OpVec
        if isinstance(front, list):
            return [self.eval(front_elem, back=back) for front_elem in front]

        elif isinstance(front, OpVec) and front.distributive:
            return front.combo_fn([self.eval(front.coeff * front_elem, back=back)
                                   for front_elem in front.oplist])
        # For now, always do this. If it's not performant, we can be more granular.
        if not isinstance(front, OperatorBase):
            front = StateFn(front, is_measurement=False)

        # Hack for speed
        if isinstance(front, StateFnDict) and isinstance(back, StateFnDict):
            sum = 0
            for (str1, str2) in itertools.product(front.primitive.keys(), back.primitive.keys()):
                bitstr1 = np.asarray(list(str1)).astype(np.bool)
                bitstr2 = np.asarray(list(str2)).astype(np.bool)

                # fix_endianness
                corrected_x_bits = self.primitive.x[::-1]
                corrected_z_bits = self.primitive.z[::-1]

                x_factor = np.logical_xor(bitstr1, bitstr2) == corrected_x_bits
                z_factor = 1 - 2 * np.logical_and(bitstr1, corrected_z_bits)
                y_factor = np.sqrt(1 - 2 * np.logical_and(corrected_x_bits, corrected_z_bits) + 0j)
                sum += self.coeff * np.product(x_factor * z_factor * y_factor) * \
                    front.primitive[str1] * front.coeff * back.primitive[str2] * back.coeff
            return sum

        new_front = None
        if isinstance(front, StateFnDict):
            new_dict = {}
            corrected_x_bits = self.primitive.x[::-1]
            corrected_z_bits = self.primitive.z[::-1]

            for bstr, v in front.primitive.items():
                bitstr = np.asarray(list(bstr)).astype(np.bool)
                new_b_str = np.logical_xor(bitstr, corrected_x_bits)
                new_str = ''.join(map(str, 1 * new_b_str))
                z_factor = np.product(1 - 2 * np.logical_and(bitstr, corrected_z_bits))
                y_factor = np.product(np.sqrt(1 - 2 * np.logical_and(corrected_x_bits,
                                                                     corrected_z_bits) + 0j))
                new_dict[new_str] = (v * z_factor * y_factor) + new_dict.get(new_str, 0)
            new_front = StateFn(new_dict, coeff=self.coeff * front.coeff)
        elif isinstance(front, StateFn):
            if front.is_measurement:
                raise ValueError('Operator composed with a measurement is undefined.')
            elif isinstance(front, StateFnVector):
                # new_front = self.eval(front.to_matrix())
                new_front = StateFnVector(np.dot(self.to_matrix(), front.to_matrix()))
            elif isinstance(front, StateFnOperator):
                new_front = StateFnOperator(OpPrimitive(self.adjoint().to_matrix() @
                                                        front.to_matrix() @
                                                        self.to_matrix()))
        elif isinstance(front, OpPauli):
            new_front = np.diag(self.compose(front).to_matrix())

        elif isinstance(front, OperatorBase):
            comp = self.to_matrix() @ front.to_matrix()
            if len(comp.shape) == 1:
                new_front = comp
            elif len(comp.shape) == 2:
                new_front = np.diag(comp)
            else:
                # Last ditch, TODO figure out what to actually do here.
                new_front = self.compose(front).reduce.eval()

        if back:
            if not isinstance(back, StateFn):
                back = StateFn(back, is_measurement=True)
            return back.eval(new_front)
        else:
            return new_front

    def exp_i(self):
        # if only one qubit is significant, we can perform the evolution
        corrected_x = self.primitive.x[::-1]
        corrected_z = self.primitive.z[::-1]
        # pylint: disable=import-outside-toplevel
        sig_qubits = np.logical_or(corrected_x, corrected_z)
        if np.sum(sig_qubits) == 0:
            # e^I is just a global phase, but we can keep track of it! Should we?
            # For now, just return identity
            return OpPauli(self.primitive)
        if np.sum(sig_qubits) == 1:
            sig_qubit_index = sig_qubits.tolist().index(True)
            # Y rotation
            if corrected_x[sig_qubit_index] and corrected_z[sig_qubit_index]:
                rot_op = OpPrimitive(RYGate(self.coeff))
            elif corrected_z[sig_qubit_index]:
                rot_op = OpPrimitive(RZGate(self.coeff))
            elif corrected_x[sig_qubit_index]:
                rot_op = OpPrimitive(RXGate(self.coeff))

            from .. import I
            left_pad = I.kronpower(sig_qubit_index)
            right_pad = I.kronpower(self.num_qubits - sig_qubit_index - 1)
            # Need to use overloaded operators here in case left_pad == I^0
            return left_pad ^ rot_op ^ right_pad
        else:
            from qiskit.aqua.operators import OpEvolution
            return OpEvolution(self)

    def __hash__(self):
        # Need this to be able to easily construct AbelianGraphs
        return id(self)

    def commutes(self, other_op):
        if not isinstance(other_op, OpPauli):
            return False
        # Don't use compose because parameters will break this
        self_bits = self.primitive.z.astype(int) + 2 * self.primitive.x.astype(int)
        other_bits = other_op.primitive.z.astype(int) + 2 * other_op.primitive.x.astype(int)
        return all((self_bits * other_bits) * (self_bits - other_bits) == 0)
