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

from qiskit.quantum_info import Statevector
from qiskit.circuit import ParameterExpression

from ..operator_base import OperatorBase
from . import StateFn
from ..combo_operators import ListOp


class VectorStateFn(StateFn):
    """ A class for representing state functions and measurements.

    State functions are defined to be complex functions over a single binary string
    (as compared to an operator,
    which is defined as a function over two binary strings, or a function taking a
    binary function to another
    binary function). This function may be called by the eval() method.

    Measurements are defined to be functionals over StateFns, taking them to real values.
    Generally, this real value
    is interpreted to represent the probability of some classical state (binary string)
    being observed from a
    probabilistic or quantum system represented by a StateFn. This leads to the equivalent
    definition, which is that
    a measurement m is a function over binary strings producing StateFns, such that the
    probability of measuring
    a given binary string b from a system with StateFn f is equal to the inner product
    between f and m(b).

    NOTE: State functions here are not restricted to wave functions,
    as there is no requirement of normalization.
    """

    # TODO maybe break up into different classes for different fn definition primitives
    # TODO allow normalization somehow?
    def __init__(self,
                 primitive: Union[list, np.ndarray, Statevector] = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 is_measurement: bool = False) -> None:
        """
        Args:
            primitive: The operator primitive being wrapped.
            coeff: A coefficient by which to multiply the state function
            is_measurement: Whether the StateFn is a measurement operator
        """
        # Lists and Numpy arrays representing statevectors are stored
        # in Statevector objects for easier handling.
        if isinstance(primitive, (np.ndarray, list)):
            primitive = Statevector(primitive)

        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)

    def get_primitives(self) -> set:
        """ Return a set of strings describing the primitives contained in the Operator """
        return {'Vector'}

    @property
    def num_qubits(self) -> int:
        return len(self.primitive.dims())

    def add(self, other: OperatorBase) -> OperatorBase:
        """ Addition. Overloaded by + in OperatorBase. """
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over statefns with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        # Right now doesn't make sense to add a StateFn to a Measurement
        if isinstance(other, VectorStateFn) and self.is_measurement == other.is_measurement:
            # Covers MatrixOperator, Statevector and custom.
            return VectorStateFn((self.coeff * self.primitive).add(other.primitive * other.coeff),
                                 is_measurement=self._is_measurement)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .. import SummedOp
        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return VectorStateFn(self.primitive.conjugate(),
                             coeff=np.conj(self.coeff),
                             is_measurement=(not self.is_measurement))

    def tensor(self, other: OperatorBase) -> OperatorBase:
        """ Tensor product
        Note: You must be conscious of Qiskit's big-endian bit printing convention.
        Meaning, Plus.tensor(Zero)
        produces a |+⟩ on qubit 0 and a |0⟩ on qubit 1, or |+⟩⨂|0⟩,
        but would produce a QuantumCircuit like
        |0⟩--
        |+⟩--
        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.
        """
        # TODO accept primitives directly in addition to PrimitiveOp?

        if isinstance(other, VectorStateFn):
            return StateFn(self.primitive.tensor(other.primitive),
                           coeff=self.coeff * other.coeff,
                           is_measurement=self.is_measurement)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .. import TensoredOp
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return numpy matrix of density operator, warn if more than 16 qubits
        to force the user to set
        massive=True if they want such a large matrix. Generally big methods
        like this should require the use of a
        converter, but in this case a convenience method for quick hacking and
        access to classical tools is
        appropriate. """

        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError(
                'to_matrix will return an exponentially large matrix,'
                ' in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        return self.primitive.to_operator().data * self.coeff

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        """
        NOTE: THIS DOES NOT RETURN A DENSITY MATRIX, IT RETURNS A CLASSICAL
        MATRIX CONTAINING THE QUANTUM OR CLASSICAL
        VECTOR REPRESENTING THE EVALUATION OF THE STATE FUNCTION ON EACH BINARY BASIS STATE.
        DO NOT ASSUME THIS IS
        IS A NORMALIZED QUANTUM OR CLASSICAL PROBABILITY VECTOR.
        If we allowed this to return a density matrix,
        then we would need to change the definition of composition to
        be ~Op @ StateFn @ Op for those cases,
        whereas by this methodology we can ensure that composition always means Op @ StateFn.

        Return numpy vector of state vector, warn if more than 16 qubits to force the user to set
        massive=True if they want such a large vector. Generally big methods
        like this should require the use of a
        converter, but in this case a convenience method for
        quick hacking and access to classical tools is
        appropriate.
        Returns:
            np.ndarray: vector of state vector
        Raises:
            ValueError: invalid parameters.
        """

        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_vector will return an exponentially large vector, in this case {0} elements.'
                ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        vec = self.primitive.data * self.coeff

        return vec if not self.is_measurement else vec.reshape(1, -1)

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        return self

    def __str__(self) -> str:
        """Overload str() """
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format('VectorStateFn' if not self.is_measurement
                                   else 'MeasurementVector', prim_str)
        else:
            return "{}({}) * {}".format('VectorStateFn' if not self.is_measurement
                                        else 'MeasurementVector',
                                        prim_str,
                                        self.coeff)

    # pylint: disable=too-many-return-statements
    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError(
                'Cannot compute overlap with StateFn or Operator if not Measurement. Try taking '
                'sf.adjoint() first to convert to measurement.')

        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn([self.eval(front.coeff * front_elem)
                                   for front_elem in front.oplist])

        if not isinstance(front, OperatorBase):
            front = StateFn(front)

        # pylint: disable=cyclic-import,import-outside-toplevel
        from . import DictStateFn, OperatorStateFn
        if isinstance(front, DictStateFn):
            return sum([v * self.primitive.data[int(b, 2)] * front.coeff
                        for (b, v) in front.primitive.items()]) * self.coeff

        if isinstance(front, VectorStateFn):
            # Need to extract the element or np.array([1]) is returned.
            return np.dot(self.to_matrix(), front.to_matrix())[0]

        if isinstance(front, OperatorStateFn):
            return front.adjoint().eval(self.primitive) * self.coeff

        return front.adjoint().eval(self.adjoint().primitive).adjoint() * self.coeff

    def sample(self,
               shots: int = 1024,
               massive: bool = False,
               reverse_endianness: bool = False) -> dict:
        """ Sample the state function as a normalized probability distribution. Returns dict of
        bitstrings in order of probability, with values being probability. """
        deterministic_counts = self.primitive.to_counts()
        # Don't need to square because to_counts already does.
        probs = np.array(list(deterministic_counts.values()))
        unique, counts = np.unique(np.random.choice(list(deterministic_counts.keys()),
                                                    size=shots,
                                                    p=(probs / sum(probs))),
                                   return_counts=True)
        counts = dict(zip(unique, counts))
        if reverse_endianness:
            scaled_dict = {bstr[::-1]: (prob / shots) for (bstr, prob) in counts.items()}
        else:
            scaled_dict = {bstr: (prob / shots) for (bstr, prob) in counts.items()}
        return dict(sorted(scaled_dict.items(), key=lambda x: x[1], reverse=True))
