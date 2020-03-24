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


import numpy as np

from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ParameterExpression

from ..operator_base import OperatorBase


class StateFn(OperatorBase):
    """ A class for representing state functions and measurements.

    State functions are defined to be complex functions over a single binary
    string (as compared to an operator,
    which is defined as a function over two binary strings, or a function
    taking a binary function to another
    binary function). This function may be called by the eval() method.

    Measurements are defined to be functionals over StateFns, taking them to
    real values. Generally, this real value
    is interpreted to represent the probability of some classical state
    (binary string) being observed from a
    probabilistic or quantum system represented by a StateFn. This leads to the
    equivalent definition, which is that
    a measurement m is a function over binary strings producing StateFns, such
    that the probability of measuring
    a given binary string b from a system with StateFn f is equal to the inner
    product between f and m(b).

    NOTE: State functions here are not restricted to wave functions, as there is
    no requirement of normalization.
    """

    @staticmethod
    # pylint: disable=unused-argument,inconsistent-return-statements
    def __new__(cls, primitive=None, coeff=1.0, is_measurement=False):
        """ A factory method to produce the correct type of StateFn subclass
        based on the primitive passed in. Primitive, coeff, and is_measurement arguments
        are passed into subclass's init() as-is automatically by new()."""

        # Prevents infinite recursion when subclasses are created
        if cls.__name__ != 'StateFn':
            return super().__new__(cls)

        # pylint: disable=cyclic-import,import-outside-toplevel
        if isinstance(primitive, (str, dict, Result)):
            from . import StateFnDict
            return StateFnDict.__new__(StateFnDict)

        if isinstance(primitive, (list, np.ndarray, Statevector)):
            from . import StateFnVector
            return StateFnVector.__new__(StateFnVector)

        if isinstance(primitive, (QuantumCircuit, Instruction)):
            from . import StateFnCircuit
            return StateFnCircuit.__new__(StateFnCircuit)

        if isinstance(primitive, OperatorBase):
            from . import StateFnOperator
            return StateFnOperator.__new__(StateFnOperator)

    # TODO allow normalization somehow?
    def __init__(self, primitive, coeff=1.0, is_measurement=False):
        """
        Args:
            primitive(str, dict, OperatorBase, Result, np.ndarray, list)
            coeff(int, float, complex): A coefficient by which to multiply the state
        """
        self._primitive = primitive
        self._is_measurement = is_measurement
        self._coeff = coeff

    @property
    def primitive(self):
        """ returns primitive """
        return self._primitive

    @property
    def coeff(self):
        """ returns coeff """
        return self._coeff

    @property
    def is_measurement(self):
        """ return if is measurement """
        return self._is_measurement

    def get_primitives(self):
        """ Return a set of strings describing the primitives contained in the Operator """
        raise NotImplementedError

    @property
    def num_qubits(self):
        raise NotImplementedError

    def add(self, other):
        """ Addition. Overloaded by + in OperatorBase. """
        raise NotImplementedError

    def neg(self):
        """ Negate. Overloaded by - in OperatorBase. """
        return self.mul(-1.0)

    def adjoint(self):
        raise NotImplementedError

    def equals(self, other):
        """ Evaluate Equality. Overloaded by == in OperatorBase. """
        if not isinstance(other, type(self)) or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive
        # Will return NotImplementedError if not supported

    def mul(self, scalar):
        """ Scalar multiply. Overloaded by * in OperatorBase.

        Doesn't multiply Statevector until to_matrix() or to_vector() is
        called to keep things lazy and avoid big
        copies.
        TODO figure out if this is a bad idea.
         """
        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))

        return self.__class__(self.primitive,
                              coeff=self.coeff * scalar,
                              is_measurement=self.is_measurement)

    def kron(self, other):
        """ Kron
        Note: You must be conscious of Qiskit's big-endian bit printing
        convention. Meaning, Plus.kron(Zero)
        produces a |+⟩ on qubit 0 and a |0⟩ on qubit 1, or |+⟩⨂|0⟩, but
        would produce a QuantumCircuit like
        |0⟩--
        |+⟩--
        Because Terra prints circuits and results with qubit 0
        at the end of the string or circuit.
        """
        raise NotImplementedError

    def kronpower(self, other):
        """ Kron with Self Multiple Times """
        if not isinstance(other, int) or other <= 0:
            raise TypeError('Kronpower can only take positive int arguments')
        temp = StateFn(self.primitive,
                       coeff=self.coeff,
                       is_measurement=self.is_measurement)
        for _ in range(other - 1):
            temp = temp.kron(self)
        return temp

    def _check_zero_for_composition_and_expand(self, other):
        new_self = self
        # pylint: disable=import-outside-toplevel
        if not self.num_qubits == other.num_qubits:
            from qiskit.aqua.operators import Zero
            if self == StateFn({'0': 1}, is_measurement=True):
                # Zero is special - we'll expand it to the correct qubit number.
                new_self = StateFn('0' * self.num_qubits, is_measurement=True)
            elif other == Zero:
                # Zero is special - we'll expand it to the correct qubit number.
                other = StateFn('0' * self.num_qubits)
            else:
                raise ValueError(
                    'Composition is not defined over Operators of different dimensions, {} and {}, '
                    'respectively.'.format(self.num_qubits, other.num_qubits))

        return new_self, other

    def to_matrix(self, massive=False):
        """ Return vector representing StateFn evaluated on each basis state.
        Must be overridden by child classes."""
        raise NotImplementedError

    def to_density_matrix(self, massive=False):
        """ Return matrix representing product of StateFn evaluated on pairs of basis states.
        Must be overridden by child classes."""
        raise NotImplementedError

    def compose(self, other):
        """ Composition (Linear algebra-style, right-to-left) is not well

        defined for States in the binary function
        model. However, it is well defined for measurements.
        """
        # TODO maybe allow outers later to produce density operators or projectors, but not yet.
        if not self.is_measurement:
            raise ValueError(
                'Composition with a Statefunction in the first operand is not defined.')

        new_self, other = self._check_zero_for_composition_and_expand(other)
        # TODO maybe include some reduction here in the subclasses - vector and Op, op and Op, etc.
        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.operators import OpCircuit

        if self.primitive == {'0' * self.num_qubits: 1.0} and isinstance(other, OpCircuit):
            # Returning StateFnCircuit
            return StateFn(other.primitive, is_measurement=self.is_measurement,
                           coeff=self.coeff * other.coeff)

        from qiskit.aqua.operators import OpComposition
        return OpComposition([new_self, other])

    def power(self, other):
        """ Compose with Self Multiple Times, undefined for StateFns. """
        raise ValueError('Composition power over Statefunctions or Measurements is not defined.')

    # def to_density_matrix(self, massive=False):
    #     """ Return numpy matrix of density operator, warn if more than 16
    #     qubits to force the user to set
    #     massive=True if they want such a large matrix. Generally big methods
    #     like this should require the use of a
    #     converter, but in this case a convenience method for quick hacking
    #     and access to classical tools is
    #     appropriate. """
    #     raise NotImplementedError

    # def to_matrix(self, massive=False):
    #     """
    #     NOTE: THIS DOES NOT RETURN A DENSITY MATRIX, IT RETURNS A CLASSICAL MATRIX CONTAINING
    #     THE QUANTUM OR CLASSICAL
    #     VECTOR REPRESENTING THE EVALUATION OF THE STATE FUNCTION ON EACH BINARY BASIS
    #     STATE. DO NOT ASSUME THIS IS
    #     IS A NORMALIZED QUANTUM OR CLASSICAL PROBABILITY VECTOR. If we allowed
    #     this to return a density matrix,
    #     then we would need to change the definition of composition to be ~Op @
    #     StateFn @ Op for those cases,
    #     whereas by this methodology we can ensure that composition always
    #     means Op @ StateFn.
    #
    #     Return numpy vector of state vector, warn if more than 16 qubits to force the user to set
    #     massive=True if they want such a large vector. Generally big methods like this
    #     should require the use of a
    #     converter, but in this case a convenience method for quick hacking
    #     and access to classical tools is
    #     appropriate. """
    #     raise NotImplementedError

    def __str__(self):
        """Overload str() """
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format('StateFunction' if not self.is_measurement
                                   else 'Measurement', self.coeff)
        else:
            return "{}({}) * {}".format('StateFunction' if not self.is_measurement
                                        else 'Measurement',
                                        self.coeff,
                                        prim_str)

    def __repr__(self):
        """Overload str() """
        return "StateFn({}, coeff={}, is_measurement={})".format(repr(self.primitive),
                                                                 self.coeff, self.is_measurement)

    # def print_details(self):
    #     """ print details """
    #     raise NotImplementedError

    def eval(self, front=None, back=None):
        """ Evaluate the State function given a basis string, dict, or state (if measurement). """
        return NotImplementedError

    # TODO
    # def sample(self, shots):
    #     """ Sample the state function as a normalized probability distribution."""
    #     raise NotImplementedError

    def bind_parameters(self, param_dict):
        """ bind parameters """
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                # pylint: disable=import-outside-toplevel
                from ..operator_combos.op_vec import OpVec
                return OpVec([self.bind_parameters(param_dict) for param_dict in unrolled_dict])
            coeff_param = list(self.coeff.parameters)[0]
            if coeff_param in unrolled_dict:
                # TODO what do we do about complex?
                value = unrolled_dict[coeff_param]
                param_value = float(self.coeff.bind({coeff_param: value}))
        return self.__class__(self.primitive, is_measurement=self.is_measurement, coeff=param_value)

    # Try collapsing primitives where possible. Nothing to collapse here.
    def reduce(self):
        # TODO replace IZ paulis with dict here?
        return self

    # Recurse into StateFn's operator with a converter if primitive is an operator.
    def traverse(self, convert_fn, coeff=None):
        """ Apply the convert_fn to each node in the oplist. """
        return StateFn(convert_fn(self.primitive),
                       coeff=coeff or self.coeff, is_measurement=self.is_measurement)

    def to_matrix_op(self, massive=False):
        """ Return a StateFnVector for this StateFn. """
        # pylint: disable=import-outside-toplevel
        from .state_fn_vector import StateFnVector
        return StateFnVector(self.to_matrix(massive=massive), is_measurement=self.is_measurement)
