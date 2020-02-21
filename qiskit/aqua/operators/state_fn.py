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
    """ A class for representing state functions over binary strings, which are equally defined to be
    1) A complex function over a single binary string (as compared to an operator, which is defined as a function
    over two binary strings).
    2) An Operator with one parameter of its evaluation function always fixed. For example, if we fix one parameter in
    the eval function of a Matrix-defined operator to be '000..0', the state function is defined by the vector which
    is the first column of the matrix (or rather, an index function over this vector). A circuit-based operator with
    one parameter fixed at '000...0' can be interpreted simply as the quantum state prepared by composing the
    circuit with the |000...0⟩ state.

    NOTE: This state function is not restricted to wave functions, as there is no requirement of normalization.

    This object is essentially defined by the operators it holds in the primitive property.
    """

    # TODO maybe break up into different classes for different fn definition primitives
    # NOTE: We call this density but we don't enforce normalization!!
    # TODO allow normalization somehow?
    def __init__(self, primitive, coeff=1.0):
        # TODO change name from primitive to something else
        """
        Args:
            primitive(str, dict, OperatorBase, np.ndarray, list)
            coeff(int, float, complex): A coefficient by which to multiply the state
        """
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
            # self._primitive = {bstr[::-1]: shots/sum(counts.values()) for (bstr, shots) in counts.items()}

        # TODO: Should we only allow correctly shaped vectors, e.g. vertical? Should we reshape to make contrast with
        #  measurement more accurate?
        elif isinstance(primitive, (np.ndarray, list)):
            self._primitive = Statevector(primitive)

        # TODO figure out custom callable later
        # if isinstance(self.primitive, callable):
        #     self._fixed_param = '0'

        self._coeff = coeff

    @property
    def primitive(self):
        return self._primitive

    @property
    def coeff(self):
        return self._coeff

    def get_primitives(self):
        """ Return a set of primitives in the StateFn """
        if isinstance(self.primitive, dict):
            return {'Dict'}
        elif isinstance(self.primitive, Statevector):
            return {'Vector'}
        if isinstance(self.primitive, OperatorBase):
            return self.primitive.get_primitives()
        else:
            # Includes 'Pauli'
            return {self.primitive.__class__.__name__}

    @property
    def num_qubits(self):
        # If the primitive is lookup of bitstrings, we define all missing strings to have a function value of
        # zero.
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
        if isinstance(other, StateFn):
            if isinstance(self.primitive, type(other.primitive)) and \
                    self.primitive == other.primitive:
                return StateFn(self.primitive, coeff=self.coeff + other.coeff)
            # Covers MatrixOperator and custom.
            elif isinstance(self.primitive, type(other.primitive)) and \
                    hasattr(self.primitive, 'add'):
                return self.primitive.add(other.primitive)

        return OpSum([self, other])

    def neg(self):
        """ Negate. Overloaded by - in OperatorBase. """
        return self.mul(-1.0)

    def adjoint(self):
        # TODO
        # return Measurement(self)
        pass

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
        return StateFn(self.primitive, coeff=self.coeff * scalar)

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
            return StateFn(new_dict, coeff=self.coeff * other.coeff)
            # TODO double check coeffs logic

        # Both Operators
        elif isinstance(self.primitive, OperatorBase) and isinstance(other.primitive, OperatorBase):
            return StateFn(self.primitive.kron(other.primitive), coeff=self.coeff * other.coeff)

        # Both Statevectors
        elif isinstance(self_primitive, Statevector) and isinstance(other_primitive, Statevector):
            return StateFn(self.primitive.tensor(other.primitive), coeff=self.coeff * other.coeff)

        # User custom kron-able primitive - Identical to Pauli above for now, but maybe remove deepcopy later
        elif isinstance(self.primitive, type(other.primitive)) and hasattr(self.primitive, 'kron'):
            sf_copy = copy.deepcopy(other.primitive)
            return StateFn(self.primitive.kron(sf_copy), coeff=self.coeff * other.coeff)

        else:
            return OpKron([self, other])

    def kronpower(self, other):
        """ Kron with Self Multiple Times """
        if not isinstance(other, int) or other <= 0:
            raise TypeError('Kronpower can only take positive int arguments')
        temp = StateFn(self.primitive, coeff=self.coeff)
        for i in range(other-1):
            temp = temp.kron(self)
        return temp

    def compose(self, other):
        """ State composition (Linear algebra-style, right-to-left) is not well defined in the binary function model.
        """
        # TODO maybe allow outers later to produce density operators or projectors, but not yet.
        raise ValueError('Composition with a Statefunctions in the first operand is not defined.')

    def power(self, other):
        """ Compose with Self Multiple Times, undefined for StateFns. """
        raise ValueError('Composition with a Statefunctions in the first operand is not defined.')

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
        VECTOR REPRESENTING THE EVALUATION OF THE STATE FUNCTION ON EACH BASIS STATE. DO NOT ASSUME THIS IS
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

        # Dict - return diagonal (real values, not complex), not rank 1 decomposition!
        if isinstance(self.primitive, dict):
            states = int(2 ** self.num_qubits)
            probs = np.zeros(states)
            for k, v in self.primitive.items():
                probs[int(k, 2)] = v
                # probs[int(k[::-1], 2)] = v
                # Note, we need to reverse the bitstring to extract an int ordering
            return probs * self.coeff

        # Operator - return diagonal (real values, not complex), not rank 1 decomposition!
        elif isinstance(self.primitive, OperatorBase):
            return np.diag(self.primitive.to_matrix()) * self.coeff

        # Statevector - Return complex values, not reals
        elif isinstance(self.primitive, Statevector):
            return self.primitive.data * self.coeff

        # User custom matrix-able primitive
        elif hasattr(self.primitive, 'to_matrix'):
            return self.primitive.to_matrix() * self.coeff

        else:
            raise NotImplementedError

    # TODO print Instructions nicely...
    def __str__(self):
        """Overload str() """
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return prim_str
        else:
            return "{} * |{}⟩".format(self.coeff, prim_str)

    def __repr__(self):
        """Overload str() """
        return "StateFn({}, coeff={}".format(repr(self.primitive), self.coeff)

    def print_details(self):
        """ print details """
        raise NotImplementedError

    def eval(self, front=None, back=None):
        # Validate bitstring: re.fullmatch(rf'[01]{{{0}}}', val1)

        # TODO decide whether to allow val2 to be used / default to val2 = val1 if None, or throw an error if it's
        #  provided, or return 0 if not val1 == val2 for diagonal types.
        if not back:
            back = front

        # If the primitive is lookup of bitstrings, we define all missing strings to have a function value of
        # zero.
        elif isinstance(self.primitive, dict):
            if front == back:
                return self.primitive.get(front, 0) * self.coeff
            else:
                return 0

        elif isinstance(self.primitive, OperatorBase):
            return self.primitive.eval(val1=front, val2=back) * self.coeff

        elif isinstance(self.primitive, Statevector):
            if front == back:
                index1 = int(front, 2)
                return self.primitive.data[index1] * self.coeff
            else:
                return 0

        elif hasattr(self.primitive, 'eval'):
            return self.primitive.eval(val1=front, val2=back)

    # TODO
    def sample(self, shots):
        """ Sample the statefunction as a normalized probability distribution."""
        raise NotImplementedError
