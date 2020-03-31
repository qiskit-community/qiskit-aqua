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

from typing import Union, Optional
import logging
import numpy as np

from qiskit import QuantumCircuit, BasicAer, execute
from qiskit.extensions.standard import IGate
from qiskit.circuit import Instruction, ParameterExpression

from ..operator_base import OperatorBase
from ..combo_operators import SummedOp, ComposedOp, TensoredOp
from .primitive_op import PrimitiveOp

logger = logging.getLogger(__name__)


class CircuitOp(PrimitiveOp):
    """ Class for Wrapping Circuit Primitives

    Note that all mathematical methods are not in-place, meaning that they return a
    new object, but the underlying primitives are not copied.

    """

    def __init__(self,
                 primitive: Union[Instruction, QuantumCircuit] = None,
                 coeff: Optional[Union[int, float, complex,
                                       ParameterExpression]] = 1.0) -> None:
        """
        Args:
            primitive (Instruction, QuantumCircuit): The operator primitive being wrapped.
            coeff (int, float, complex): A coefficient multiplying the primitive
        Raises:
            TypeError: invalid parameters.
        """
        if isinstance(primitive, QuantumCircuit):
            primitive = primitive.to_instruction()

        if not isinstance(primitive, Instruction):
            raise TypeError('CircuitOp can only be instantiated with '
                            'Instruction, not {}'.format(type(primitive)))

        super().__init__(primitive, coeff=coeff)

    def get_primitives(self) -> set:
        """ Return a set of strings describing the primitives contained in the Operator """
        return {'QuantumCircuit'}

    # TODO replace with proper alphabets later?
    @property
    def num_qubits(self) -> int:
        return self.primitive.num_qubits

    # TODO change to *other to efficiently handle lists?
    def add(self, other: OperatorBase) -> OperatorBase:
        """ Addition. Overloaded by + in OperatorBase. """
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over operators with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        if isinstance(other, CircuitOp) and self.primitive == other.primitive:
            return CircuitOp(self.primitive, coeff=self.coeff + other.coeff)

        # Covers all else.
        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        """ Return operator adjoint (conjugate transpose). Overloaded by ~ in OperatorBase. """
        return CircuitOp(self.primitive.inverse(), coeff=np.conj(self.coeff))

    def equals(self, other: OperatorBase) -> bool:
        """ Evaluate Equality. Overloaded by == in OperatorBase. """
        if not isinstance(other, PrimitiveOp) \
                or not isinstance(self.primitive, type(other.primitive)) \
                or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive
        # Will return NotImplementedError if not supported

    # TODO change to *other to handle lists? How aggressively to handle pairwise business?
    def tensor(self, other: OperatorBase) -> OperatorBase:
        """ Tensor product
        Note: You must be conscious of Qiskit's big-endian bit printing
        convention. Meaning, X.tensor(Y)
        produces an X on qubit 0 and an Y on qubit 1, or X⨂Y, but would produce a
        QuantumCircuit which looks like
        -[Y]-
        -[X]-
        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.
        """
        # TODO accept primitives directly in addition to PrimitiveOp?
        # pylint: disable=cyclic-import,import-outside-toplevel
        from . import PauliOp
        if isinstance(other, PauliOp):
            from qiskit.aqua.operators.converters import PauliToInstruction
            other = CircuitOp(PauliToInstruction().convert_pauli(other.primitive),
                              coeff=other.coeff)

        if isinstance(other, CircuitOp):
            new_qc = QuantumCircuit(self.num_qubits + other.num_qubits)
            # NOTE!!! REVERSING QISKIT ENDIANNESS HERE
            new_qc.append(other.primitive, new_qc.qubits[0:other.primitive.num_qubits])
            new_qc.append(self.primitive, new_qc.qubits[other.primitive.num_qubits:])
            # TODO Fix because converting to dag just to append is nuts
            # TODO Figure out what to do with cbits?
            return CircuitOp(new_qc.decompose().to_instruction(), coeff=self.coeff * other.coeff)

        return TensoredOp([self, other])

    # TODO change to *other to efficiently handle lists?
    def compose(self, other: OperatorBase) -> OperatorBase:
        """ Operator Composition (Linear algebra-style, right-to-left)

        Note: You must be conscious of Quantum Circuit vs. Linear Algebra ordering
        conventions. Meaning, X.compose(Y)
        produces an X∘Y on qubit 0, but would produce a QuantumCircuit which looks like
        -[Y]-[X]-
        Because Terra prints circuits with the initial state at the left side of the circuit.
        """
        # TODO accept primitives directly in addition to PrimitiveOp?

        other = self._check_zero_for_composition_and_expand(other)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from ..operator_globals import Zero
        from ..state_functions import CircuitStateFn
        if other == Zero ^ self.num_qubits:
            return CircuitStateFn(self.primitive, coeff=self.coeff)

        from . import PauliOp
        if isinstance(other, PauliOp):
            from qiskit.aqua.operators.converters import PauliToInstruction
            other = CircuitOp(PauliToInstruction().convert_pauli(other.primitive),
                              coeff=other.coeff)

        if isinstance(other, (CircuitOp, CircuitStateFn)):
            new_qc = QuantumCircuit(self.num_qubits)
            new_qc.append(other.primitive, qargs=range(self.num_qubits))
            new_qc.append(self.primitive, qargs=range(self.num_qubits))
            # TODO Fix because converting to dag just to append is nuts
            # TODO Figure out what to do with cbits?
            new_qc = new_qc.decompose()
            if isinstance(other, CircuitStateFn):
                return CircuitStateFn(new_qc.to_instruction(),
                                      is_measurement=other.is_measurement,
                                      coeff=self.coeff * other.coeff)
            else:
                return CircuitOp(new_qc.to_instruction(), coeff=self.coeff * other.coeff)

        return ComposedOp([self, other])

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return numpy matrix of operator, warn if more than 16 qubits
        to force the user to set massive=True if
        they want such a large matrix. Generally big methods like this
        should require the use of a converter,
        but in this case a convenience method for quick hacking and
        access to classical tools is appropriate. """

        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError(
                'to_matrix will return an exponentially large matrix,'
                ' in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        qc = QuantumCircuit(self.primitive.num_qubits)
        # NOTE: not reversing qubits!!
        # qc.append(self.primitive, qargs=range(self.primitive.num_qubits)[::-1])
        qc.append(self.primitive, qargs=range(self.primitive.num_qubits))
        unitary_backend = BasicAer.get_backend('unitary_simulator')
        unitary = execute(qc, unitary_backend, optimization_level=0).result().get_unitary()
        return unitary * self.coeff

    def __str__(self) -> str:
        """Overload str() """
        qc = self.reduce().to_circuit()
        prim_str = str(qc.draw(output='text'))
        if self.coeff == 1.0:
            return prim_str
        else:
            return "{} * {}".format(self.coeff, prim_str)

    def bind_parameters(self, param_dict: dict) -> OperatorBase:
        param_value = self.coeff
        qc = self.primitive
        if isinstance(self.coeff, ParameterExpression) or self.primitive.params:
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                # pylint: disable=import-outside-toplevel
                from ..combo_operators.list_op import ListOp
                return ListOp([self.bind_parameters(param_dict) for param_dict in unrolled_dict])
            if self.coeff in unrolled_dict:
                # TODO what do we do about complex?
                param_value = float(self.coeff.bind(unrolled_dict[self.coeff]))
            if all(param in unrolled_dict for param in self.primitive.params):
                qc = self.to_circuit().decompose().bind_parameters(param_dict)
        return self.__class__(qc, coeff=param_value)

    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        """ A square binary Operator can be defined as a function over two binary
        strings of equal length. This
        method returns the value of that function for a given pair of binary strings.
        For more information,
        see the eval method in operator_base.py.

        Notice that Pauli evals will always return 0 for Paulis with X or Y terms if val1 == val2.
        This is why we must
        convert to a {Z,I}^n Pauli basis to take "averaging" style expectations
        (e.g. PauliExpectation).
        """

        # pylint: disable=import-outside-toplevel
        from ..state_functions import CircuitStateFn
        from ..combo_operators import ListOp
        from .pauli_op import PauliOp

        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn([self.eval(front.coeff * front_elem)
                                   for front_elem in front.oplist])

        # Composable with circuit
        if isinstance(front, (PauliOp, CircuitStateFn, CircuitOp)):
            return self.compose(front)

        return self.to_matrix_op().eval(front=front)

    def to_circuit(self) -> QuantumCircuit:
        """ Convert CircuitOp to circuit """
        qc = QuantumCircuit(self.num_qubits)
        qc.append(self.primitive, qargs=range(self.primitive.num_qubits))
        return qc

    # Warning - modifying immutable object!!
    def reduce(self) -> OperatorBase:
        if self.primitive._definition is not None:
            for i, inst_context in enumerate(self.primitive._definition):
                [gate, _, _] = inst_context
                if isinstance(gate, IGate):
                    del self.primitive._definition[i]
        return self
