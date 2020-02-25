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

""" An Object to represent State Functions constructed from Operators """


import numpy as np
import re
from functools import reduce
import itertools

from qiskit.quantum_info import Statevector
from qiskit.result import Result

from qiskit.aqua.operators.operator_base import OperatorBase


class StateFn(OperatorBase):
    """ A class for representing state functions and measurements.

    State functions are defined to be complex functions over a single binary string (as compared to an operator,
    which is defined as a function over two binary strings, or a function taking a binary function to another
    binary function). This function may be called by the eval() method.

    Measurements are defined to be functionals over StateFns, taking them to real values. Generally, this real value
    is interpreted to represent the probability of some classical state (binary string) being observed from a
    probabilistic or quantum system represented by a StateFn. This leads to the equivalent definition, which is that
    a measurement m is a function over binary strings producing StateFns, such that the probability of measuring
    a given binary string b from a system with StateFn f is equal to the inner product between f and m(b).

    NOTE: State functions here are not restricted to wave functions, as there is no requirement of normalization.
    """

    # TODO maybe break up into different classes for different fn definition primitives
    # TODO allow normalization somehow?
    def __init__(self, primitive, coeff=1.0, is_measurement=False):
        """
        Args:
            primitive(str, dict, OperatorBase, Result, np.ndarray, list)
            coeff(int, float, complex): A coefficient by which to multiply the state
        """
        self._is_measurement = is_measurement
        self._coeff = coeff

        # If the initial density is a string, treat this as a density dict with only a single basis state.
        if isinstance(primitive, str):
            self._primitive = {primitive: 1}

        # If the initial density is set to a counts dict, Statevector, or an operator, treat it as a density operator,
        # where the eval function is equal to eval(my_str, my_str), e.g. a lookup along the diagonal.
        elif isinstance(primitive, (dict, OperatorBase, Statevector)):
            self._primitive = primitive

        # NOTE:
        # 1) This is not the same as passing in the counts dict directly, as this will convert the shot numbers to
        # probabilities, whereas passing in the counts dict will not.
        # 2) This will extract counts for both shot and statevector simulations. To use the statevector,
        # simply pass in the statevector.
        # 3) This will only extract the first result.
        if isinstance(primitive, Result):
            counts = primitive.get_counts()
            self._primitive = {bstr: shots/sum(counts.values()) for (bstr, shots) in counts.items()}

        # Lists and Numpy arrays representing statevectors are stored in Statevector objects for easier handling.
        elif isinstance(primitive, (np.ndarray, list)):
            self._primitive = Statevector(primitive)

        # TODO figure out custom callable later
        # if isinstance(self.primitive, callable):

    @property
    def primitive(self):
        return self._primitive

    @property
    def coeff(self):
        return self._coeff

    @property
    def is_measurement(self):
        return self._is_measurement

    def get_primitives(self):
        """ Return a set of strings describing the primitives contained in the Operator """
        if isinstance(self.primitive, dict):
            return {'Dict'}
        elif isinstance(self.primitive, Statevector):
            return {'Vector'}
        if isinstance(self.primitive, OperatorBase):
            return self.primitive.get_primitives()
        else:
            return {self.primitive.__class__.__name__}

    @property
    def num_qubits(self):
        if isinstance(self.primitive, dict):
            return len(list(self.primitive.keys())[0])

        elif isinstance(self.primitive, Statevector):
            return len(self.primitive.dims())

        elif isinstance(self.primitive, OperatorBase):
            return self.primitive.num_qubits

    def add(self, other):
        """ Addition. Overloaded by + in OperatorBase. """
        if not self.num_qubits == other.num_qubits:
            raise ValueError('Sum over statefns with different numbers of qubits, {} and {}, is not well '
                             'defined'.format(self.num_qubits, other.num_qubits))
        # Right now doesn't make sense to add a StateFn to a Measurement
        if isinstance(other, StateFn) and self.is_measurement == other.is_measurement:
            if isinstance(self.primitive, type(other.primitive)) and self.primitive == other.primitive:
                return StateFn(self.primitive,
                               coeff=self.coeff + other.coeff,
                               is_measurement=self.is_measurement)
            # Covers MatrixOperator, Statevector and custom.
            elif isinstance(self.primitive, type(other.primitive)) and \
                    hasattr(self.primitive, 'add'):
                # Also assumes scalar multiplication is available
                return StateFn((self.coeff * self.primitive).add(other.primitive * other.coeff),
                               is_measurement=self._is_measurement)
            elif isinstance(self.primitive, dict) and isinstance(other.primitive):
                new_dict = {b: (v * self.coeff) + (other.primitive.get(b, 0) * other.coeff)
                            for (b, v) in self.primitive.items()}
                new_dict.update({b: v*other.coeff for (b, v) in other.primitive.items() if b not in self.primitive})
                return StateFn(new_dict, is_measurement=self._is_measurement)

        from . import OpSum
        return OpSum([self, other])

    def neg(self):
        """ Negate. Overloaded by - in OperatorBase. """
        return self.mul(-1.0)

    def adjoint(self):
        if isinstance(self.primitive, Statevector):
            prim_adjoint = self.primitive.conjugate()
        elif isinstance(self.primitive, OperatorBase):
            prim_adjoint = self.primitive.adjoint()
        elif isinstance(self.primitive, dict):
            prim_adjoint = {b: np.conj(v) for (b, v) in self.primitive.items()}
        else:
            prim_adjoint = self.primitive
        return StateFn(prim_adjoint,
                       coeff=self.coeff,
                       is_measurement=(not self.is_measurement))

    def equals(self, other):
        """ Evaluate Equality. Overloaded by == in OperatorBase. """
        if not isinstance(other, StateFn) \
                or not isinstance(self.primitive, type(other.primitive)) \
                or not self.coeff == other.coeff:
            return False
        return self.primitive == other.primitive
        # Will return NotImplementedError if not supported

    def mul(self, scalar):
        """ Scalar multiply. Overloaded by * in OperatorBase.

        Doesn't multiply Statevector until to_matrix() or to_vector() is called to keep things lazy and avoid big
        copies.
        TODO figure out if this is a bad idea.
         """
        if not isinstance(scalar, (int, float, complex)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))
        return StateFn(self.primitive,
                       coeff=self.coeff * scalar,
                       is_measurement=self.is_measurement)

    def kron(self, other):
        """ Kron
        Note: You must be conscious of Qiskit's big-endian bit printing convention. Meaning, Plus.kron(Zero)
        produces a |+⟩ on qubit 0 and a |0⟩ on qubit 1, or |+⟩⨂|0⟩, but would produce a QuantumCircuit like
        |0⟩--
        |+⟩--
        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.
        """
        # TODO accept primitives directly in addition to OpPrimitive?

        # Both dicts
        if isinstance(self.primitive, dict) and isinstance(other.primitive, dict):
            new_dict = {k1+k2: v1*v2 for ((k1, v1,), (k2, v2)) in
                        itertools.product(self.primitive.items(), other.primitive.items())}
            return StateFn(new_dict,
                           coeff=self.coeff * other.coeff,
                           is_measurement=self.is_measurement)
            # TODO double check coeffs logic

        # Both Operators
        elif isinstance(self.primitive, OperatorBase) and isinstance(other.primitive, OperatorBase):
            return StateFn(self.primitive.kron(other.primitive),
                           coeff=self.coeff * other.coeff,
                           is_measurement=self.is_measurement)

        # Both Statevectors
        elif isinstance(self_primitive, Statevector) and isinstance(other_primitive, Statevector):
            return StateFn(self.primitive.tensor(other.primitive),
                           coeff=self.coeff * other.coeff,
                           is_measurement=self.is_measurement)

        # User custom kron-able primitive - Identical to Pauli above for now, but maybe remove deepcopy later
        elif isinstance(self.primitive, type(other.primitive)) and hasattr(self.primitive, 'kron'):
            sf_copy = copy.deepcopy(other.primitive)
            return StateFn(self.primitive.kron(sf_copy),
                           coeff=self.coeff * other.coeff,
                           is_measurement=self.is_measurement)

        else:
            from . import OpKron
            return OpKron([self, other])

    def kronpower(self, other):
        """ Kron with Self Multiple Times """
        if not isinstance(other, int) or other <= 0:
            raise TypeError('Kronpower can only take positive int arguments')
        temp = StateFn(self.primitive,
                       coeff=self.coeff,
                       is_measurement=self.is_measurement)
        for i in range(other-1):
            temp = temp.kron(self)
        return temp

    def compose(self, other):
        """ Composition (Linear algebra-style, right-to-left) is not well defined for States in the binary function
        model. However, it is well defined for measurements.
        """
        # TODO maybe allow outers later to produce density operators or projectors, but not yet.
        if not self.is_measurement:
            raise ValueError('Composition with a Statefunctions in the first operand is not defined.')
        # TODO: Handle this for measurement @ something else.

        new_self = self
        if not self.num_qubits == other.num_qubits:
            if self.primitive == StateFn({'0': 1}, is_measurement=True):
                # Zero is special - we'll expand it to the correct qubit number.
                new_self = StateFn('0' * self.num_qubits, is_measurement=True)
            else:
                raise ValueError('Composition is not defined over Operators of different dimensions, {} and {}, '
                                 'respectively.'.format(self.num_qubits, other.num_qubits))

        from . import OpComposition
        return OpComposition([new_self, other])

    def power(self, other):
        """ Compose with Self Multiple Times, undefined for StateFns. """
        raise ValueError('Composition power over Statefunctions or Measurements is not defined.')

    def to_density_matrix(self, massive=False):
        """ Return numpy matrix of density operator, warn if more than 16 qubits to force the user to set
        massive=True if they want such a large matrix. Generally big methods like this should require the use of a
        converter, but in this case a convenience method for quick hacking and access to classical tools is
        appropriate. """

        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError('to_matrix will return an exponentially large matrix, in this case {0}x{0} elements.'
                             ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        # Dict
        if isinstance(self.primitive, dict):
            return self.to_matrix() * np.eye(states) * self.coeff

        # Operator
        elif isinstance(self.primitive, OperatorBase):
            return self.primitive.to_matrix() * self.coeff

        # Statevector
        elif isinstance(self.primitive, Statevector):
            return self.primitive.to_operator().data * self.coeff

        # User custom matrix-able primitive
        elif hasattr(self.primitive, 'to_matrix'):
            return self.primitive.to_matrix() * self.coeff

        else:
            raise NotImplementedError

    def to_matrix(self, massive=False):
        """
        NOTE: THIS DOES NOT RETURN A DENSITY MATRIX, IT RETURNS A CLASSICAL MATRIX CONTAINING THE QUANTUM OR CLASSICAL
        VECTOR REPRESENTING THE EVALUATION OF THE STATE FUNCTION ON EACH BINARY BASIS STATE. DO NOT ASSUME THIS IS
        IS A NORMALIZED QUANTUM OR CLASSICAL PROBABILITY VECTOR. If we allowed this to return a density matrix,
        then we would need to change the definition of composition to be ~Op @ StateFn @ Op for those cases,
        whereas by this methodology we can ensure that composition always means Op @ StateFn.

        Return numpy vector of state vector, warn if more than 16 qubits to force the user to set
        massive=True if they want such a large vector. Generally big methods like this should require the use of a
        converter, but in this case a convenience method for quick hacking and access to classical tools is
        appropriate. """

        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError('to_vector will return an exponentially large vector, in this case {0} elements.'
                             ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        # Dict
        if isinstance(self.primitive, dict):
            states = int(2 ** self.num_qubits)
            probs = np.zeros(states)
            for k, v in self.primitive.items():
                probs[int(k, 2)] = v
                # probs[int(k[::-1], 2)] = v
                # TODO Remove comment: Note, we need to reverse the bitstring to extract an int ordering
            vec = probs * self.coeff

        # Operator - return diagonal (real values, not complex), not rank 1 decomposition (statevector)!
        elif isinstance(self.primitive, OperatorBase):
            mat = self.primitive.to_matrix()
            if isinstance(mat, list):
                vec = [np.diag(op) * self.coeff for op in mat]
            else:
                vec = np.diag(mat) * self.coeff

        # Statevector - Return complex values, not reals
        elif isinstance(self.primitive, Statevector):
            vec = self.primitive.data * self.coeff

        # User custom matrix-able primitive
        elif hasattr(self.primitive, 'to_matrix'):
            vec = self.primitive.to_matrix() * self.coeff

        else:
            raise NotImplementedError

        # Reshape for measurements so np.dot still works for composition.
        if isinstance(vec, list):
            return vec if not self.is_measurement else [op.reshape(1, -1) for op in vec]
        return vec if not self.is_measurement else vec.reshape(1, -1)

    def __str__(self):
        """Overload str() """
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format('StateFunction' if not self.is_measurement else 'Measurement', self.coeff)
        else:
            return "{}({}) * {}".format('StateFunction' if not self.is_measurement else 'Measurement',
                                        self.coeff,
                                        prim_str)

    def __repr__(self):
        """Overload str() """
        return "StateFn({}, coeff={}, is_measurement={}".format(repr(self.primitive), self.coeff, self.is_measurement)

    def print_details(self):
        """ print details """
        raise NotImplementedError

    def eval(self, other=None):
        # Validate bitstring: re.fullmatch(rf'[01]{{{0}}}', val1)

        if isinstance(other, str):
            other = {str: 1}

        # If the primitive is a lookup of bitstrings, we define all missing strings to have a function value of
        # zero.
        if isinstance(self.primitive, dict) and isinstance(other, dict):
            return sum([v * other.get(b, 0) for (b, v) in self.primitive.items()]) * self.coeff

        if not self.is_measurement and isinstance(other, OperatorBase):
            raise ValueError('Cannot compute overlap with StateFn or Operator if not Measurement. Try taking '
                             'sf.adjoint() first to convert to measurement.')

        # All remaining possibilities only apply when self.is_measurement is True

        if isinstance(other, StateFn):
            if isinstance(other.primitive, OperatorBase):
                if isinstance(self.primitive, OperatorBase):
                    # Both are density matrices, need to compose and trace
                    return np.trace(self.to_matrix() @ other.to_matrix())
                else:
                    return self.eval(other.primitive).eval(self.adjoint()) * self.coeff
            elif isinstance(other.primitive, (Statevector, dict)):
                return self.eval(other.primitive) * other.coeff

        if isinstance(self.primitive, dict):
            if isinstance(other, Statevector):
                return sum([v * other.data[int(b, 2)] * np.conj(other.data[int(b, 2)])
                            for (b, v) in self.primitive.items()]) * self.coeff
            if isinstance(other, OperatorBase):
                # TODO Wrong, need to eval from both sides
                return other.eval(self.primitive).adjoint()

        # Measurement is specified as Density matrix.
        if isinstance(self.primitive, OperatorBase):
            if isinstance(other, OperatorBase):
                # Compose the other Operator to self's measurement density matrix
                return StateFn(other.adjoint().compose(self.primitive).compose(other),
                               coeff=self.coeff,
                               is_measurement=True)
            else:
                # Written this way to be able to handle many types of other (at least dict and Statevector).
                return self.primitive.eval(other).adjoint().eval(other) * self.coeff

        elif isinstance(self.primitive, Statevector):
            if isinstance(other, dict):
                return sum([v * self.primitive.data[int(b, 2)] for (b, v) in other.items()]) * self.coeff
            elif isinstance(other, Statevector):
                return np.dot(self.primitive.data, other.data) * self.coeff

        # TODO figure out what to actually do here.
        else:
            return self.sample(1024)

    # TODO
    def sample(self, shots):
        """ Sample the statefunction as a normalized probability distribution."""
        raise NotImplementedError

    # Try collapsing primitives where possible. Nothing to collapse here.
    def reduce(self):
        # TODO replace IZ paulis with dict here?
        return self

    # Recurse into StateFn's operator with a converter if primitive is an operator.
    def traverse(self, convert_fn, coeff=None):
        """ Apply the convert_fn to each node in the oplist. """
        return StateFn(convert_fn(self.primitive), coeff=coeff or self.coeff, is_measurement=self.is_measurement)
