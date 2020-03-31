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

from qiskit.circuit import ParameterExpression

from ..operator_base import OperatorBase
from .state_fn import StateFn
from ..combo_operators import ListOp, SummedOp


# pylint: disable=invalid-name

class OperatorStateFn(StateFn):
    """ A class for representing state functions and measurements.

    State functions are defined to be complex functions over a single binary string (as
    compared to an operator, which is defined as a function over two binary strings, or
    a function taking a binary function to another binary function). This function may be
    called by the eval() method.

    Measurements are defined to be functionals over StateFns, taking them to real values.
    Generally, this real value is interpreted to represent the probability of some classical
    state (binary string) being observed from a probabilistic or quantum system represented
    by a StateFn. This leads to the equivalent definition, which is that a measurement m is a
    function over binary strings producing StateFns, such that the probability of measuring
    a given binary string b from a system with StateFn f is equal to the inner product between
    f and m(b).

    NOTE: State functions here are not restricted to wave functions,
    as there is no requirement of normalization.
    """

    # TODO maybe break up into different classes for different fn definition primitives
    # TODO allow normalization somehow?
    def __init__(self,
                 primitive: Union[OperatorBase] = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 is_measurement: bool = False) -> None:
        """
        Args:
            primitive: The operator primitive being wrapped.
            coeff: A coefficient by which to multiply the state function
            is_measurement: Whether the StateFn is a measurement operator
        """

        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)

    def get_primitives(self) -> set:
        """ Return a set of strings describing the primitives contained in the Operator """
        return self.primitive.get_primitives()

    @property
    def num_qubits(self) -> int:
        return self.primitive.num_qubits

    def add(self, other: OperatorBase) -> OperatorBase:
        """ Addition. Overloaded by + in OperatorBase. """
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over statefns with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        # Right now doesn't make sense to add a StateFn to a Measurement
        if isinstance(other, OperatorStateFn) and self.is_measurement == other.is_measurement:
            if isinstance(self.primitive.primitive, type(other.primitive.primitive)) and \
                    self.primitive == other.primitive:
                return StateFn(self.primitive,
                               coeff=self.coeff + other.coeff,
                               is_measurement=self.is_measurement)
            # Covers MatrixOperator, Statevector and custom.
            elif isinstance(other, OperatorStateFn):
                # Also assumes scalar multiplication is available
                return OperatorStateFn(
                    (self.coeff * self.primitive).add(other.primitive * other.coeff),
                    is_measurement=self._is_measurement)

        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return OperatorStateFn(self.primitive.adjoint(),
                               coeff=np.conj(self.coeff),
                               is_measurement=(not self.is_measurement))

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

        if isinstance(other, OperatorStateFn):
            return StateFn(self.primitive.tensor(other.primitive),
                           coeff=self.coeff * other.coeff,
                           is_measurement=self.is_measurement)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .. import TensoredOp
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return numpy matrix of density operator, warn if more than 16 qubits
        to force the user to set
        massive=True if they want such a large matrix. Generally big methods like
        this should require the use of a
        converter, but in this case a convenience method for quick hacking and
        access to classical tools is
        appropriate. """

        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError(
                'to_matrix will return an exponentially large matrix,'
                ' in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        # TODO handle list case
        return self.primitive.to_matrix() * self.coeff

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """ Return a MatrixOp for this operator. """
        return OperatorStateFn(self.primitive.to_matrix_op(massive=massive) * self.coeff,
                               is_measurement=self.is_measurement)

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        """
        NOTE: THIS DOES NOT RETURN A DENSITY MATRIX, IT RETURNS A CLASSICAL MATRIX
        CONTAINING THE QUANTUM OR CLASSICAL
        VECTOR REPRESENTING THE EVALUATION OF THE STATE FUNCTION ON EACH BINARY
        BASIS STATE. DO NOT ASSUME THIS IS
        IS A NORMALIZED QUANTUM OR CLASSICAL PROBABILITY VECTOR. If we allowed
        this to return a density matrix,
        then we would need to change the definition of composition to be ~Op @
        StateFn @ Op for those cases,
        whereas by this methodology we can ensure that composition always means Op @ StateFn.

        Return numpy vector of state vector, warn if more than 16 qubits to force the user to set
        massive=True if they want such a large vector. Generally big methods like
        this should require the use of a
        converter, but in this case a convenience method for quick hacking and
        access to classical tools is appropriate.
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

        # Operator - return diagonal (real values, not complex),
        # not rank 1 decomposition (statevector)!
        mat = self.primitive.to_matrix()
        # TODO change to sum of eigenvectors?

        # ListOp primitives can return lists of matrices (or trees for nested ListOps),
        # so we need to recurse over the
        # possible tree.
        def diag_over_tree(t):
            if isinstance(t, list):
                return [diag_over_tree(o) for o in t]
            else:
                vec = np.diag(t) * self.coeff
                # Reshape for measurements so np.dot still works for composition.
                return vec if not self.is_measurement else vec.reshape(1, -1)

        return diag_over_tree(mat)

    def __str__(self) -> str:
        """Overload str() """
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format('StateFunction' if not self.is_measurement
                                   else 'Measurement', prim_str)
        else:
            return "{}({}) * {}".format(
                'StateFunction' if not self.is_measurement else 'Measurement',
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

        if not isinstance(front, OperatorBase):
            front = StateFn(front)

        if isinstance(self.primitive, ListOp) and self.primitive.distributive:
            evals = [OperatorStateFn(op, coeff=self.coeff, is_measurement=self.is_measurement).eval(
                front) for op in self.primitive.oplist]
            return self.primitive.combo_fn(evals)

        # Need an ListOp-specific carve-out here to make sure measurement over an ListOp doesn't
        # produce two-dimensional ListOp from composing from both sides of primitive.
        # Can't use isinstance because this would include subclasses.
        # pylint: disable=unidiomatic-typecheck
        if type(front) == ListOp:
            return front.combo_fn([self.eval(front.coeff * front_elem)
                                   for front_elem in front.oplist])

        return front.adjoint().eval(self.primitive.eval(front))

    def sample(self,
               shots: int = 1024,
               massive: bool = False,
               reverse_endianness: bool = False) -> dict:
        """ Sample the state function as a normalized probability distribution. Returns dict of
        bitstrings in order of probability, with values being probability. """
        raise NotImplementedError
