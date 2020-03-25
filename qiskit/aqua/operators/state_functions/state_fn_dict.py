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

import itertools
import numpy as np

from qiskit.result import Result

from ..operator_base import OperatorBase
from . import StateFn
from ..operator_combos import OpVec


class StateFnDict(StateFn):
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
    def __init__(self, primitive, coeff=1.0, is_measurement=False):
        """
        Args:
            primitive(str, dict, OperatorBase, Result, np.ndarray, list)
            coeff(int, float, complex): A coefficient by which to multiply the state
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
            # NOTE: Need to square root to take Pauli measurements!
            primitive = {bstr: (shots / sum(counts.values()))**.5 for
                         (bstr, shots) in counts.items()}

        if not isinstance(primitive, dict):
            raise TypeError(
                'StateFnDict can only be instantiated with dict, '
                'string, or Qiskit Result, not {}'.format(type(primitive)))

        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)

    def get_primitives(self):
        """ Return a set of strings describing the primitives contained in the Operator """
        return {'Dict'}

    @property
    def num_qubits(self):
        return len(list(self.primitive.keys())[0])

    def add(self, other):
        """ Addition. Overloaded by + in OperatorBase. """
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over statefns with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        # Right now doesn't make sense to add a StateFn to a Measurement
        if isinstance(other, StateFnDict) and self.is_measurement == other.is_measurement:
            # TODO add compatibility with vector and Operator?
            if self.primitive == other.primitive:
                return StateFnDict(self.primitive,
                                   coeff=self.coeff + other.coeff,
                                   is_measurement=self.is_measurement)
            else:
                new_dict = {b: (v * self.coeff) + (other.primitive.get(b, 0) * other.coeff)
                            for (b, v) in self.primitive.items()}
                new_dict.update({b: v * other.coeff for (b, v) in other.primitive.items()
                                 if b not in self.primitive})
                return StateFn(new_dict, is_measurement=self._is_measurement)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from qiskit.aqua.operators import OpSum
        return OpSum([self, other])

    def adjoint(self):
        return StateFnDict({b: np.conj(v) for (b, v) in self.primitive.items()},
                           coeff=np.conj(self.coeff),
                           is_measurement=(not self.is_measurement))

    def kron(self, other):
        """ Kron
        Note: You must be conscious of Qiskit's big-endian bit printing convention.
        Meaning, Plus.kron(Zero)
        produces a |+⟩ on qubit 0 and a |0⟩ on qubit 1, or |+⟩⨂|0⟩,
        but would produce a QuantumCircuit like
        |0⟩--
        |+⟩--
        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.
        """
        # TODO accept primitives directly in addition to OpPrimitive?

        # Both dicts
        if isinstance(other, StateFnDict):
            new_dict = {k1 + k2: v1 * v2 for ((k1, v1,), (k2, v2)) in
                        itertools.product(self.primitive.items(), other.primitive.items())}
            return StateFn(new_dict,
                           coeff=self.coeff * other.coeff,
                           is_measurement=self.is_measurement)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from qiskit.aqua.operators import OpKron
        return OpKron([self, other])

    def to_density_matrix(self, massive=False):
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

    def to_matrix(self, massive=False):
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

    def __str__(self):
        """Overload str() """
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format('StateFnDict' if not self.is_measurement
                                   else 'MeasurementDict', prim_str)
        else:
            return "{}({}) * {}".format('StateFnDict' if not self.is_measurement
                                        else 'MeasurementDict',
                                        prim_str,
                                        self.coeff)

    # pylint: disable=too-many-return-statements
    def eval(self, front=None):

        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError(
                'Cannot compute overlap with StateFn or Operator if not Measurement. Try taking '
                'sf.adjoint() first to convert to measurement.')

        if isinstance(front, OpVec) and front.distributive:
            return front.combo_fn([self.eval(front.coeff * front_elem)
                                   for front_elem in front.oplist])

        # For now, always do this. If it's not performant, we can be more granular.
        if not isinstance(front, OperatorBase):
            front = StateFn(front)

        # If the primitive is a lookup of bitstrings,
        # we define all missing strings to have a function value of
        # zero.
        if isinstance(front, StateFnDict):
            return sum([v * front.primitive.get(b, 0) for (b, v) in
                        self.primitive.items()]) * self.coeff * front.coeff

        # All remaining possibilities only apply when self.is_measurement is True

        # pylint: disable=cyclic-import,import-outside-toplevel
        from . import StateFnVector
        if isinstance(front, StateFnVector):
            # TODO does it need to be this way for measurement?
            # return sum([v * front.primitive.data[int(b, 2)] *
            # np.conj(front.primitive.data[int(b, 2)])
            return sum([v * front.primitive.data[int(b, 2)]
                        for (b, v) in self.primitive.items()]) * self.coeff

        from . import StateFnOperator
        if isinstance(front, StateFnOperator):
            return front.adjoint().eval(self.adjoint())

        # All other OperatorBases go here
        return front.adjoint().eval(self.adjoint().primitive).adjoint() * self.coeff

    def sample(self, shots=1024, massive=False, reverse_endianness=False):
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
