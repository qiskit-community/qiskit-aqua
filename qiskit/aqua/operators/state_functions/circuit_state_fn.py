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

""" An Object to represent State Functions constructed from Operators """

from typing import Union
import numpy as np

from qiskit import QuantumCircuit, BasicAer, execute
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.extensions import Initialize, IGate

from ..operator_base import OperatorBase
from ..combo_operators import SummedOp
from .state_fn import StateFn


class CircuitStateFn(StateFn):
    """ A class for representing state functions and measurements.

    State functions are defined to be complex functions over a single binary string
    (as compared to an operator,
    which is defined as a function over two binary strings, or a function taking
    a binary function to another
    binary function). This function may be called by the eval() method.

    Measurements are defined to be functionals over StateFns, taking them to real values.
    Generally, this real value
    is interpreted to represent the probability of some classical state (binary string)
    being observed from a
    probabilistic or quantum system represented by a StateFn. This leads to the
    equivalent definition, which is that
    a measurement m is a function over binary strings producing StateFns, such that
     the probability of measuring
    a given binary string b from a system with StateFn f is equal to the inner
    product between f and m(b).

    NOTE: State functions here are not restricted to wave functions, as there
    is no requirement of normalization.
    """

    # TODO maybe break up into different classes for different fn definition primitives
    # TODO allow normalization somehow?
    def __init__(self,
                 primitive: Union[QuantumCircuit, Instruction] = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 is_measurement: bool = False) -> None:
        """
            Args:
                primitive: The operator primitive being wrapped.
                coeff: A coefficient by which to multiply
                 the state function.
                is_measurement: Whether the StateFn is a measurement operator.

            Raises:
                TypeError: invalid parameters.
        """
        if isinstance(primitive, QuantumCircuit):
            primitive = primitive.to_instruction()

        if not isinstance(primitive, Instruction):
            raise TypeError('CircuitStateFn can only be instantiated '
                            'with Instruction, not {}'.format(type(primitive)))

        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)

    @staticmethod
    def from_dict(density_dict: dict) -> OperatorBase:
        """ from dictionary """
        # If the dict is sparse (elements <= qubits), don't go
        # building a statevector to pass to Qiskit's
        # initializer, just create a sum.
        if len(density_dict) <= len(list(density_dict.keys())[0]):
            statefn_circuits = []
            for bstr, prob in density_dict.items():
                qc = QuantumCircuit(len(bstr))
                # NOTE: Reversing endianness!!
                for (index, bit) in enumerate(reversed(bstr)):
                    if bit == '1':
                        qc.x(index)
                sf_circuit = CircuitStateFn(qc, coeff=prob)
                statefn_circuits += [sf_circuit]
            if len(statefn_circuits) == 1:
                return statefn_circuits[0]
            else:
                return SummedOp(statefn_circuits)
        else:
            sf_dict = StateFn(density_dict)
            return CircuitStateFn.from_vector(sf_dict.to_matrix())

    @staticmethod
    def from_vector(statevector: np.ndarray) -> OperatorBase:
        """ from vector """
        normalization_coeff = np.linalg.norm(statevector)
        normalized_sv = statevector / normalization_coeff
        if not np.all(np.abs(statevector) == statevector):
            raise ValueError('Qiskit circuit Initializer cannot handle non-positive statevectors.')
        return CircuitStateFn(Initialize(normalized_sv), coeff=normalization_coeff)

    def get_primitives(self) -> set:
        """ Return a set of strings describing the primitives contained in the Operator """
        return {'QuantumCircuit'}

    @property
    def num_qubits(self) -> int:
        return self.primitive.num_qubits

    def add(self, other: OperatorBase) -> OperatorBase:
        """ Addition. Overloaded by + in OperatorBase. """
        if not self.num_qubits == other.num_qubits:
            raise ValueError('Sum over operators with different numbers of qubits, '
                             '{} and {}, is not well '
                             'defined'.format(self.num_qubits, other.num_qubits))

        if isinstance(other, CircuitStateFn) and self.primitive == other.primitive:
            return CircuitStateFn(self.primitive, coeff=self.coeff + other.coeff)

        # Covers all else.
        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return CircuitStateFn(self.primitive.inverse(),
                              coeff=np.conj(self.coeff),
                              is_measurement=(not self.is_measurement))

    def compose(self, other: OperatorBase) -> OperatorBase:
        """ Composition (Linear algebra-style, right-to-left) is not well defined
        for States in the binary function
        model. However, it is well defined for measurements.
        """
        # TODO maybe allow outers later to produce density operators or projectors, but not yet.
        if not self.is_measurement:
            raise ValueError(
                'Composition with a Statefunctions in the first operand is not defined.')

        new_self, other = self._check_zero_for_composition_and_expand(other)

        # pylint: disable=cyclic-import,import-outside-toplevel
        from qiskit.aqua.operators import CircuitOp, PauliOp

        if isinstance(other, (CircuitOp, PauliOp)):
            op_circuit_self = CircuitOp(self.primitive)

            # Avoid reimplementing compose logic
            composed_op_circs = op_circuit_self.compose(other)

            # Returning CircuitStateFn
            return CircuitStateFn(composed_op_circs.primitive,
                                  is_measurement=self.is_measurement,
                                  coeff=self.coeff * other.coeff)

        if isinstance(other, CircuitStateFn) and self.is_measurement:
            from .. import Zero
            return self.compose(CircuitOp(other.primitive,
                                          other.coeff)).compose(Zero ^ self.num_qubits)

        from qiskit.aqua.operators import ComposedOp
        return ComposedOp([new_self, other])

    def tensor(self, other: OperatorBase) -> OperatorBase:
        """ Tensor product
        Note: You must be conscious of Qiskit's big-endian bit printing convention.
        Meaning, Plus.tensor(Zero)
        produces a |+⟩ on qubit 0 and a |0⟩ on qubit 1, or |+⟩⨂|0⟩, but would produce
        a QuantumCircuit like
        |0⟩--
        |+⟩--
        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.
        """
        # TODO accept primitives directly in addition to PrimitiveOp?

        if isinstance(other, CircuitStateFn):
            new_qc = QuantumCircuit(self.num_qubits + other.num_qubits)
            # NOTE!!! REVERSING QISKIT ENDIANNESS HERE
            new_qc.append(other.primitive, new_qc.qubits[0:other.primitive.num_qubits])
            new_qc.append(self.primitive, new_qc.qubits[other.primitive.num_qubits:])
            # TODO Fix because converting to dag just to append is nuts
            # TODO Figure out what to do with cbits?
            return CircuitStateFn(new_qc.decompose().to_instruction(),
                                  coeff=self.coeff * other.coeff)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from qiskit.aqua.operators import TensoredOp
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return numpy matrix of density operator, warn if more than 16 qubits to
        force the user to set
        massive=True if they want such a large matrix. Generally big methods like this
        should require the use of a
        converter, but in this case a convenience method for quick hacking and access
        to classical tools is
        appropriate. """

        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError(
                'to_matrix will return an exponentially large matrix,'
                ' in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        # TODO handle list case
        # Rely on VectorStateFn's logic here.
        return StateFn(self.primitive.to_matrix() * self.coeff).to_density_matrix()

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        """
        NOTE: THIS DOES NOT RETURN A DENSITY MATRIX, IT RETURNS A CLASSICAL MATRIX CONTAINING
         THE QUANTUM OR CLASSICAL
        VECTOR REPRESENTING THE EVALUATION OF THE STATE FUNCTION ON EACH BINARY BASIS STATE.
        DO NOT ASSUME THIS IS
        IS A NORMALIZED QUANTUM OR CLASSICAL PROBABILITY VECTOR. If we allowed this to
        return a density matrix,
        then we would need to change the definition of composition to be ~Op @ StateFn @
        Op for those cases,
        whereas by this methodology we can ensure that composition always means Op @ StateFn.

        Return numpy vector of state vector, warn if more than 16 qubits to force the user to set
        massive=True if they want such a large vector. Generally big methods like this
        should require the use of a
        converter, but in this case a convenience method for quick hacking and access
        to classical tools is
        appropriate.
        Returns:
            np.ndarray: vector of state vector
        Raises:
            ValueError: invalid parameters.
        """

        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError(
                'to_vector will return an exponentially large vector, in this case {0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        # Need to adjoint to get forward statevector and then reverse
        if self.is_measurement:
            return np.conj(self.adjoint().to_matrix())
        qc = self.to_circuit(meas=False)
        statevector_backend = BasicAer.get_backend('statevector_simulator')
        statevector = execute(qc, statevector_backend,
                              optimization_level=0).result().get_statevector()
        return statevector * self.coeff

    def __str__(self) -> str:
        """Overload str() """
        qc = self.reduce().to_circuit()
        prim_str = str(qc.draw(output='text'))
        if self.coeff == 1.0:
            return "{}(\n{}\n)".format('StateFunction' if not self.is_measurement
                                       else 'Measurement', prim_str)
        else:
            return "{}(\n{}\n) * {}".format('StateFunction' if not self.is_measurement
                                            else 'Measurement',
                                            prim_str,
                                            self.coeff)

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
        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError(
                'Cannot compute overlap with StateFn or Operator if not Measurement. Try taking '
                'sf.adjoint() first to convert to measurement.')

        # pylint: disable=import-outside-toplevel
        from ..combo_operators import ListOp
        from ..primitive_operators import PauliOp, CircuitOp

        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn([self.eval(front.coeff * front_elem)
                                   for front_elem in front.oplist])

        # Composable with circuit
        if isinstance(front, (PauliOp, CircuitStateFn, CircuitOp)):
            new_front = self.compose(front)
            return new_front

        return self.to_matrix_op().eval(front)

    def to_circuit(self, meas: bool = False) -> QuantumCircuit:
        """ to circuit """
        if meas:
            qc = QuantumCircuit(self.num_qubits, self.num_qubits)
            qc.append(self.primitive, qargs=range(self.primitive.num_qubits))
            qc.measure(qubit=range(self.num_qubits), cbit=range(self.num_qubits))
        else:
            qc = QuantumCircuit(self.num_qubits)
            qc.append(self.primitive, qargs=range(self.primitive.num_qubits))
        # Need to decompose to unwrap instruction. TODO this is annoying, fix it
        return qc.decompose()

    # TODO specify backend?
    def sample(self,
               shots: int = 1024,
               massive: bool = False,
               reverse_endianness: bool = False) -> dict:
        """ Sample the state function as a normalized probability distribution. Returns dict of
        bitstrings in order of probability, with values being probability. """
        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError(
                'to_vector will return an exponentially large vector, in this case {0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        qc = self.to_circuit(meas=True)
        qasm_backend = BasicAer.get_backend('qasm_simulator')
        counts = execute(qc, qasm_backend, optimization_level=0, shots=shots).result().get_counts()
        if reverse_endianness:
            scaled_dict = {bstr[::-1]: (prob / shots) for (bstr, prob) in counts.items()}
        else:
            scaled_dict = {bstr: (prob / shots) for (bstr, prob) in counts.items()}
        return dict(sorted(scaled_dict.items(), key=lambda x: x[1], reverse=True))

    # Warning - modifying immutable object!!
    def reduce(self) -> OperatorBase:
        if self.primitive._definition is not None:
            for i, inst_context in enumerate(self.primitive._definition):
                [gate, _, _] = inst_context
                if isinstance(gate, IGate):
                    del self.primitive._definition[i]
        return self
