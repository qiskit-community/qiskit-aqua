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

""" Wrapping Pauli Primitives """

from typing import Union, Optional, Set
import logging
import numpy as np
from scipy.sparse import spmatrix

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, Instruction
from qiskit.quantum_info import Pauli
from qiskit.extensions.standard import RZGate, RYGate, RXGate, XGate, YGate, ZGate, IGate

from ..operator_base import OperatorBase
from .primitive_op import PrimitiveOp
from ..combo_operators.summed_op import SummedOp
from ..combo_operators.composed_op import ComposedOp
from ..combo_operators.tensored_op import TensoredOp

logger = logging.getLogger(__name__)
PAULI_GATE_MAPPING = {'X': XGate(), 'Y': YGate(), 'Z': ZGate(), 'I': IGate()}


class PauliOp(PrimitiveOp):
    """ Class for Wrapping Pauli Primitives

    Note that all mathematical methods are not in-place,
    meaning that they return a new object, but the underlying
    primitives are not copied.

    """

    def __init__(self,
                 primitive: Union[Pauli] = None,
                 coeff: Optional[Union[int, float, complex, ParameterExpression]] = 1.0) -> None:
        """
            Args:
                primitive: The operator primitive being wrapped.
                coeff: A coefficient multiplying the primitive.

            Raises:
                TypeError: invalid parameters.
        """
        if not isinstance(primitive, Pauli):
            raise TypeError(
                'PauliOp can only be instantiated with Paulis, not {}'.format(type(primitive)))
        super().__init__(primitive, coeff=coeff)

    def primitive_strings(self) -> Set[str]:
        """ Return a set of strings describing the primitives contained in the Operator """
        return {'Pauli'}

    @property
    def num_qubits(self) -> int:
        return len(self.primitive)

    def add(self, other: OperatorBase) -> OperatorBase:
        """ Addition. Overloaded by + in OperatorBase. """
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over operators with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        if isinstance(other, PauliOp) and self.primitive == other.primitive:
            return PauliOp(self.primitive, coeff=self.coeff + other.coeff)

        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        """ Return operator adjoint (conjugate transpose). Overloaded by ~ in OperatorBase. """
        return PauliOp(self.primitive, coeff=np.conj(self.coeff))

    def equals(self, other: OperatorBase) -> bool:
        """ Evaluate Equality. Overloaded by == in OperatorBase. """
        if not isinstance(other, PauliOp) or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive

    def tensor(self, other: OperatorBase) -> OperatorBase:
        """ Tensor product
        Note: You must be conscious of Qiskit's big-endian bit
        printing convention. Meaning, X.tensor(Y)
        produces an X on qubit 0 and an Y on qubit 1, or X⨂Y,
        but would produce a
        QuantumCircuit which looks like
        -[Y]-
        -[X]-
        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.
        """
        # Both Paulis
        if isinstance(other, PauliOp):
            # TODO change Pauli tensor product in Terra to have optional in place
            op_copy = Pauli(x=other.primitive.x, z=other.primitive.z)
            # NOTE!!! REVERSING QISKIT ENDIANNESS HERE
            return PauliOp(op_copy.kron(self.primitive), coeff=self.coeff * other.coeff)

        # pylint: disable=cyclic-import,import-outside-toplevel
        from .circuit_op import CircuitOp
        if isinstance(other, CircuitOp):
            return self.to_circuit_op().tensor(other)

        return TensoredOp([self, other])

    def compose(self, other: OperatorBase) -> OperatorBase:
        """ Operator Composition (Linear algebra-style, right-to-left)

        Note: You must be conscious of Quantum Circuit vs. Linear Algebra ordering
        conventions. Meaning, X.compose(Y)
        produces an X∘Y on qubit 0, but would produce a QuantumCircuit which looks like
        -[Y]-[X]-
        Because Terra prints circuits with the initial state at the left side of the circuit.
        """
        other = self._check_zero_for_composition_and_expand(other)

        # If self is identity, just return other.
        if not any(self.primitive.x + self.primitive.z):
            return other * self.coeff

        # Both Paulis
        if isinstance(other, PauliOp):
            product, phase = Pauli.sgn_prod(self.primitive, other.primitive)
            return PrimitiveOp(product, coeff=self.coeff * other.coeff * phase)

        # pylint: disable=cyclic-import,import-outside-toplevel
        from .circuit_op import CircuitOp
        from ..state_functions.circuit_state_fn import CircuitStateFn
        if isinstance(other, (CircuitOp, CircuitStateFn)):
            return self.to_circuit_op().compose(other)

        return ComposedOp([self, other])

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return numpy matrix of operator, warn if more than
        16 qubits to force the user to set massive=True if
        they want such a large matrix. Generally big methods like
        this should require the use of a converter,
        but in this case a convenience method for quick hacking
        and access to classical tools is appropriate. """

        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_matrix will return an exponentially large matrix, '
                'in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        return self.primitive.to_matrix() * self.coeff

    def to_spmatrix(self) -> spmatrix:
        """ Return scipy sparse matrix of operator. """
        return self.primitive.to_spmatrix() * self.coeff

    def __str__(self) -> str:
        """Overload str() """
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return prim_str
        else:
            return "{} * {}".format(self.coeff, prim_str)

    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
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

        if front is None:
            return self.to_matrix_op()

        # pylint: disable=import-outside-toplevel
        from ..state_functions.state_fn import StateFn
        from ..state_functions.dict_state_fn import DictStateFn
        from ..state_functions.circuit_state_fn import CircuitStateFn
        from ..combo_operators.list_op import ListOp
        from .circuit_op import CircuitOp

        new_front = None

        # For now, always do this. If it's not performant, we can be more granular.
        if not isinstance(front, OperatorBase):
            front = StateFn(front, is_measurement=False)

        if isinstance(front, ListOp) and front.distributive:
            new_front = front.combo_fn([self.eval(front.coeff * front_elem)
                                        for front_elem in front.oplist])

        elif isinstance(front, DictStateFn):
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

        elif isinstance(front, StateFn) and front.is_measurement:
            raise ValueError('Operator composed with a measurement is undefined.')

        # Composable types with PauliOp
        elif isinstance(front, (PauliOp, CircuitOp, CircuitStateFn)):
            new_front = self.compose(front)

        # Covers VectorStateFn and OperatorStateFn
        elif isinstance(front, OperatorBase):
            new_front = self.to_matrix_op().eval(front.to_matrix_op())

        return new_front

    def exp_i(self) -> OperatorBase:
        # if only one qubit is significant, we can perform the evolution
        corrected_x = self.primitive.x[::-1]
        corrected_z = self.primitive.z[::-1]
        # pylint: disable=import-outside-toplevel,no-member
        sig_qubits = np.logical_or(corrected_x, corrected_z)
        if np.sum(sig_qubits) == 0:
            # e^I is just a global phase, but we can keep track of it! Should we?
            # For now, just return identity
            return PauliOp(self.primitive)
        if np.sum(sig_qubits) == 1:
            sig_qubit_index = sig_qubits.tolist().index(True)
            # Y rotation
            if corrected_x[sig_qubit_index] and corrected_z[sig_qubit_index]:
                rot_op = PrimitiveOp(RYGate(self.coeff))
            # Z rotation
            elif corrected_z[sig_qubit_index]:
                rot_op = PrimitiveOp(RZGate(self.coeff))
            # X rotation
            elif corrected_x[sig_qubit_index]:
                rot_op = PrimitiveOp(RXGate(self.coeff))

            from ..operator_globals import I
            left_pad = I.tensorpower(sig_qubit_index)
            right_pad = I.tensorpower(self.num_qubits - sig_qubit_index - 1)
            # Need to use overloaded operators here in case left_pad == I^0
            return left_pad ^ rot_op ^ right_pad
        else:
            from ..evolutions.evolved_op import EvolvedOp
            return EvolvedOp(self)

    def __hash__(self) -> int:
        # Need this to be able to easily construct AbelianGraphs
        return id(self)

    def commutes(self, other_op) -> bool:
        """ commutes """
        if not isinstance(other_op, PauliOp):
            return False
        # Don't use compose because parameters will break this
        self_bits = self.primitive.z.astype(int) + 2 * self.primitive.x.astype(int)
        other_bits = other_op.primitive.z.astype(int) + 2 * other_op.primitive.x.astype(int)
        return all((self_bits * other_bits) * (self_bits - other_bits) == 0)

    def to_circuit(self) -> QuantumCircuit:
        """ returns a circuit constructed from this Pauli """
        # If Pauli equals identity, don't skip the IGates
        is_identity = sum(self.primitive.x + self.primitive.z) == 0

        # Note: Reversing endianness!!
        qc = QuantumCircuit(len(self.primitive))
        for q, pauli_str in enumerate(reversed(self.primitive.to_label())):
            gate = PAULI_GATE_MAPPING[pauli_str]
            if not pauli_str == 'I' or is_identity:
                qc.append(gate, qargs=[q])
        return qc

    def to_instruction(self) -> Instruction:
        # TODO just do this because performance of adding and deleting IGates doesn't matter?
        # Reduce to remove extra IGates.
        # return PrimitiveOp(self.primitive.to_instruction(), coeff=self.coeff).reduce()

        return self.to_circuit().to_instruction()
