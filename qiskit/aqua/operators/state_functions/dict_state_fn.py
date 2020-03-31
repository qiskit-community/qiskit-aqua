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
import itertools
import numpy as np
from scipy import sparse

from qiskit.result import Result
from qiskit.circuit import ParameterExpression

from ..operator_base import OperatorBase
from . import StateFn
from ..combo_operators import ListOp


class DictStateFn(StateFn):
    """ A class for representing state functions and measurements.

    State functions are defined to be complex functions over a single binary string
    (as compared to an operator,
    which is defined as a function over two binary strings, or a function taking a binary
    function to another
    binary function). This function may be called by the eval() method.

    Measurements are defined to be functionals over StateFns, taking them to real values.
    Generally, this real value
    is interpreted to represent the probability of some classical state (binary string)
    being observed from a
    probabilistic or quantum system represented by a StateFn. This leads to the
    equivalent definition, which is that
    a measurement m is a function over binary strings producing StateFns, such that
    the probability of measuring
    a given binary string b from a system with StateFn f is equal to the inner product
    between f and m(b).

    NOTE: State functions here are not restricted to wave functions, as there is
    no requirement of normalization.
    """

    # TODO maybe break up into different classes for different fn definition primitives
    # TODO allow normalization somehow?
    def __init__(self,
                 primitive: Union[str, dict, Result] = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 is_measurement: bool = False) -> None:
        """
            Args:
                primitive: The operator primitive being wrapped.
                coeff: A coefficient by which to multiply the state function.
                is_measurement: Whether the StateFn is a measurement operator.

            Raises:
                    TypeError: invalid parameters.
        """
        # If the initial density is a string, treat this as a density dict
        # with only a single basis state.
        if isinstance(primitive, str):
            primitive = {primitive: 1}

        # NOTE:
        # 1) This is not the same as passing in the counts dict directly, as this will
        # convert the shot numbers to
        # probabilities, whereas passing in the counts dict will not.
        # 2) This will extract counts for both shot and statevector simulations.
        # To use the statevector,
        # simply pass in the statevector.
        # 3) This will only extract the first result.
        if isinstance(primitive, Result):
            counts = primitive.get_counts()
            # NOTE: Need to square root to take correct Pauli measurements!
            primitive = {bstr: (shots / sum(counts.values()))**.5 for
                         (bstr, shots) in counts.items()}

        if not isinstance(primitive, dict):
            raise TypeError(
                'DictStateFn can only be instantiated with dict, '
                'string, or Qiskit Result, not {}'.format(type(primitive)))

        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)

    def get_primitives(self) -> set:
        """ Return a set of strings describing the primitives contained in the Operator """
        return {'Dict'}

    @property
    def num_qubits(self) -> int:
        return len(list(self.primitive.keys())[0])

    def add(self, other: OperatorBase) -> OperatorBase:
        """ Addition. Overloaded by + in OperatorBase. """
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over statefns with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        # Right now doesn't make sense to add a StateFn to a Measurement
        if isinstance(other, DictStateFn) and self.is_measurement == other.is_measurement:
            # TODO add compatibility with vector and Operator?
            if self.primitive == other.primitive:
                return DictStateFn(self.primitive,
                                   coeff=self.coeff + other.coeff,
                                   is_measurement=self.is_measurement)
            else:
                new_dict = {b: (v * self.coeff) + (other.primitive.get(b, 0) * other.coeff)
                            for (b, v) in self.primitive.items()}
                new_dict.update({b: v * other.coeff for (b, v) in other.primitive.items()
                                 if b not in self.primitive})
                return StateFn(new_dict, is_measurement=self._is_measurement)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from qiskit.aqua.operators import SummedOp
        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return DictStateFn({b: np.conj(v) for (b, v) in self.primitive.items()},
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

        # Both dicts
        if isinstance(other, DictStateFn):
            new_dict = {k1 + k2: v1 * v2 for ((k1, v1,), (k2, v2)) in
                        itertools.product(self.primitive.items(), other.primitive.items())}
            return StateFn(new_dict,
                           coeff=self.coeff * other.coeff,
                           is_measurement=self.is_measurement)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from qiskit.aqua.operators import TensoredOp
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return numpy matrix of density operator, warn if more than 16 qubits to
        force the user to set
        massive=True if they want such a large matrix. Generally big methods
        like this should require the use of a
        converter, but in this case a convenience method for quick
        hacking and access to classical tools is
        appropriate. """

        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError(
                'to_matrix will return an exponentially large matrix,'
                ' in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        states = int(2 ** self.num_qubits)
        return self.to_matrix() * np.eye(states) * self.coeff

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        """
        NOTE: THIS DOES NOT RETURN A DENSITY MATRIX, IT RETURNS A CLASSICAL MATRIX CONTAINING
        THE QUANTUM OR CLASSICAL
        VECTOR REPRESENTING THE EVALUATION OF THE STATE FUNCTION ON EACH BINARY BASIS STATE.
        DO NOT ASSUME THIS IS
        IS A NORMALIZED QUANTUM OR CLASSICAL PROBABILITY VECTOR. If we allowed this to return
        a density matrix,
        then we would need to change the definition of composition to
        be ~Op @ StateFn @ Op for those cases,
        whereas by this methodology we can ensure that composition always means Op @ StateFn.

        Return numpy vector of state vector, warn if more than 16 qubits to force the user to set
        massive=True if they want such a large vector. Generally big methods like this
        should require the use of a
        converter, but in this case a convenience method for quick hacking and access
        to classical tools is appropriate.
        Returns:
            np.ndarray: vector of state vector
        Raises:
            ValueError: invalid parameters.
        """

        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError(
                'to_vector will return an exponentially large vector, in this case {0} elements.'
                ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        states = int(2 ** self.num_qubits)
        # Convert vector to float.
        # TODO just take abs instead?
        probs = np.zeros(states) + 0.j
        for k, v in self.primitive.items():
            probs[int(k, 2)] = v
            # probs[int(k[::-1], 2)] = v
            # TODO Remove comment after more testing: Note, we need to
            #  reverse the bitstring to extract an int ordering
        vec = probs * self.coeff

        # Reshape for measurements so np.dot still works for composition.
        return vec if not self.is_measurement else vec.reshape(1, -1)

    def to_spmatrix(self) -> sparse.spmatrix:
        """
        Same as to_matrix, but returns csr sparse matrix.
        Returns:
            sparse.csr_matrix: vector of state vector
        Raises:
            ValueError: invalid parameters.
        """

        indices = [int(v, 2) for v in self.primitive.keys()]
        vals = np.array(list(self.primitive.values())) * self.coeff
        spvec = sparse.csr_matrix((vals, (np.zeros(len(indices), dtype=int), indices)),
                                  shape=(1, 2**self.num_qubits))
        return spvec if not self.is_measurement else spvec.transpose()

    def __str__(self) -> str:
        """Overload str() """
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format('DictStateFn' if not self.is_measurement
                                   else 'MeasurementDict', prim_str)
        else:
            return "{}({}) * {}".format('DictStateFn' if not self.is_measurement
                                        else 'MeasurementDict',
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

        # For now, always do this. If it's not performant, we can be more granular.
        if not isinstance(front, OperatorBase):
            front = StateFn(front)

        # If the primitive is a lookup of bitstrings,
        # we define all missing strings to have a function value of
        # zero.
        if isinstance(front, DictStateFn):
            return sum([v * front.primitive.get(b, 0) for (b, v) in
                        self.primitive.items()]) * self.coeff * front.coeff

        # All remaining possibilities only apply when self.is_measurement is True

        # pylint: disable=cyclic-import,import-outside-toplevel
        from . import VectorStateFn
        if isinstance(front, VectorStateFn):
            # TODO does it need to be this way for measurement?
            # return sum([v * front.primitive.data[int(b, 2)] *
            # np.conj(front.primitive.data[int(b, 2)])
            return sum([v * front.primitive.data[int(b, 2)]
                        for (b, v) in self.primitive.items()]) * self.coeff

        from . import OperatorStateFn
        if isinstance(front, OperatorStateFn):
            return front.adjoint().eval(self.adjoint())

        # All other OperatorBases go here
        return front.adjoint().eval(self.adjoint().primitive).adjoint() * self.coeff

    def sample(self,
               shots: int = 1024,
               massive: bool = False,
               reverse_endianness: bool = False) -> dict:
        """ Sample the state function as a normalized probability distribution. Returns dict of
        bitstrings in order of probability, with values being probability. """
        probs = np.array(list(self.primitive.values()))**2
        unique, counts = np.unique(np.random.choice(list(self.primitive.keys()),
                                                    size=shots,
                                                    p=(probs / sum(probs))),
                                   return_counts=True)
        counts = dict(zip(unique, counts))
        if reverse_endianness:
            scaled_dict = {bstr[::-1]: (prob / shots) for (bstr, prob) in counts.items()}
        else:
            scaled_dict = {bstr: (prob / shots) for (bstr, prob) in counts.items()}
        return dict(sorted(scaled_dict.items(), key=lambda x: x[1], reverse=True))
