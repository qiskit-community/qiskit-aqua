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


from typing import Union, Optional, Callable
import numpy as np

from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ParameterExpression

from ..operator_base import OperatorBase


class StateFn(OperatorBase):
    """ A class for representing state functions and measurements.

    State functions are defined to be complex functions over a single binary string (as
    compared to an operator, which is defined as a function over two binary strings, or a
    function taking a binary function to another binary function). This function may be
    called by the eval() method.

    Measurements are defined to be functionals over StateFns, taking them to real values.
    Generally, this real value is interpreted to represent the probability of some classical
    state (binary string) being observed from a probabilistic or quantum system represented
    by a StateFn. This leads to the equivalent definition, which is that a measurement m is
    a function over binary strings producing StateFns, such that the probability of measuring
    a given binary string b from a system with StateFn f is equal to the inner
    product between f and m(b).

    NOTE: State functions here are not restricted to wave functions, as there is
    no requirement of normalization.
    """

    @staticmethod
    # pylint: disable=unused-argument,inconsistent-return-statements
    def __new__(cls,
                primitive: Union[str, dict, Result,
                                 list, np.ndarray, Statevector,
                                 QuantumCircuit, Instruction,
                                 OperatorBase] = None,
                coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                is_measurement: bool = False) -> OperatorBase:
        """ A factory method to produce the correct type of StateFn subclass
        based on the primitive passed in. Primitive, coeff, and is_measurement arguments
        are passed into subclass's init() as-is automatically by new()."""

        # Prevents infinite recursion when subclasses are created
        if cls.__name__ != StateFn.__name__:
            return super().__new__(cls)

        # pylint: disable=cyclic-import,import-outside-toplevel
        if isinstance(primitive, (str, dict, Result)):
            from . import DictStateFn
            return DictStateFn.__new__(DictStateFn)

        if isinstance(primitive, (list, np.ndarray, Statevector)):
            from . import VectorStateFn
            return VectorStateFn.__new__(VectorStateFn)

        if isinstance(primitive, (QuantumCircuit, Instruction)):
            from . import CircuitStateFn
            return CircuitStateFn.__new__(CircuitStateFn)

        if isinstance(primitive, OperatorBase):
            from . import OperatorStateFn
            return OperatorStateFn.__new__(OperatorStateFn)

    # TODO allow normalization somehow?
    def __init__(self,
                 primitive: Union[str, dict, Result,
                                  list, np.ndarray, Statevector,
                                  QuantumCircuit, Instruction,
                                  OperatorBase] = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 is_measurement: bool = False) -> None:
        """
        Args:
            primitive: The operator primitive being wrapped.
            coeff: A coefficient by which to multiply the state function.
            is_measurement: Whether the StateFn is a measurement operator
        """
        self._primitive = primitive
        self._is_measurement = is_measurement
        self._coeff = coeff

    @property
    def primitive(self):
        """ returns primitive """
        return self._primitive

    @property
    def coeff(self) -> Union[int, float, complex, ParameterExpression]:
        """ returns coeff """
        return self._coeff

    @property
    def is_measurement(self) -> bool:
        """ return if is measurement """
        return self._is_measurement

    def get_primitives(self) -> set:
        """ Return a set of strings describing the primitives contained in the Operator """
        raise NotImplementedError

    @property
    def num_qubits(self) -> int:
        raise NotImplementedError

    def add(self, other: OperatorBase) -> OperatorBase:
        """ Addition. Overloaded by + in OperatorBase. """
        raise NotImplementedError

    def neg(self) -> OperatorBase:
        """ Negate. Overloaded by - in OperatorBase. """
        return self.mul(-1.0)

    def adjoint(self) -> OperatorBase:
        raise NotImplementedError

    def equals(self, other: OperatorBase) -> bool:
        """ Evaluate Equality. Overloaded by == in OperatorBase. """
        if not isinstance(other, type(self)) or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive
        # Will return NotImplementedError if not supported

    def mul(self, scalar: Union[int, float, complex, ParameterExpression]) -> OperatorBase:
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

    def tensor(self, other: OperatorBase) -> OperatorBase:
        """ Tensor product
        Note: You must be conscious of Qiskit's big-endian bit printing
        convention. Meaning, Plus.tensor(Zero)
        produces a |+⟩ on qubit 0 and a |0⟩ on qubit 1, or |+⟩⨂|0⟩, but
        would produce a QuantumCircuit like
        |0⟩--
        |+⟩--
        Because Terra prints circuits and results with qubit 0
        at the end of the string or circuit.
        """
        raise NotImplementedError

    def tensorpower(self, other: int) -> Union[OperatorBase, int]:
        """ Tensor product with Self Multiple Times """
        if not isinstance(other, int) or other <= 0:
            raise TypeError('Tensorpower can only take positive int arguments')
        temp = StateFn(self.primitive,
                       coeff=self.coeff,
                       is_measurement=self.is_measurement)
        for _ in range(other - 1):
            temp = temp.tensor(self)
        return temp

    def _check_zero_for_composition_and_expand(self, other: OperatorBase) \
            -> (OperatorBase, OperatorBase):
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

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return numpy vector representing StateFn evaluated on each basis state. Warn if more
        than 16 qubits to force having to set massive=True if such a large vector is desired.
        Must be overridden by child classes.

        NOTE: THIS DOES NOT RETURN A DENSITY MATRIX, IT RETURNS A CLASSICAL MATRIX CONTAINING
        THE QUANTUM OR CLASSICAL VECTOR REPRESENTING THE EVALUATION OF THE STATE FUNCTION ON
        EACH BINARY BASIS STATE. DO NOT ASSUME THIS IS IS A NORMALIZED QUANTUM OR CLASSICAL
        PROBABILITY VECTOR. If we allowed this to return a density matrix, then we would need
        to change the definition of composition to be ~Op @ StateFn @ Op for those cases,
        whereas by this methodology we can ensure that composition always means Op @ StateFn.
        """
        raise NotImplementedError

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return matrix representing product of StateFn evaluated on pairs of basis states.
        Must be overridden by child classes."""
        raise NotImplementedError

    def compose(self, other: OperatorBase) -> OperatorBase:
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
        from qiskit.aqua.operators import CircuitOp

        if self.primitive == {'0' * self.num_qubits: 1.0} and isinstance(other, CircuitOp):
            # Returning CircuitStateFn
            return StateFn(other.primitive, is_measurement=self.is_measurement,
                           coeff=self.coeff * other.coeff)

        from qiskit.aqua.operators import ComposedOp
        return ComposedOp([new_self, other])

    def power(self, other: int) -> OperatorBase:
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

    def __str__(self) -> str:
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

    def __repr__(self) -> str:
        """Overload str() """
        return "StateFn({}, coeff={}, is_measurement={})".format(repr(self.primitive),
                                                                 self.coeff, self.is_measurement)

    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        """ Evaluate the State function given a basis string, dict, or state (if measurement). """
        raise NotImplementedError

    def bind_parameters(self, param_dict: dict) -> OperatorBase:
        """ bind parameters """
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                # pylint: disable=import-outside-toplevel
                from ..combo_operators.list_op import ListOp
                return ListOp([self.bind_parameters(param_dict) for param_dict in unrolled_dict])
            coeff_param = list(self.coeff.parameters)[0]
            if coeff_param in unrolled_dict:
                # TODO what do we do about complex?
                value = unrolled_dict[coeff_param]
                param_value = float(self.coeff.bind({coeff_param: value}))
        return self.__class__(self.primitive, is_measurement=self.is_measurement, coeff=param_value)

    # Try collapsing primitives where possible. Nothing to collapse here.
    def reduce(self) -> OperatorBase:
        return self

    # Recurse into StateFn's operator with a converter if primitive is an operator.
    def traverse(self,
                 convert_fn: Callable,
                 coeff: Optional[Union[int, float, complex,
                                       ParameterExpression]] = None) -> OperatorBase:
        """ Apply the convert_fn to each node in the oplist. """
        return StateFn(convert_fn(self.primitive),
                       coeff=coeff or self.coeff, is_measurement=self.is_measurement)

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """ Return a VectorStateFn for this StateFn. """
        # pylint: disable=import-outside-toplevel
        from .vector_state_fn import VectorStateFn
        return VectorStateFn(self.to_matrix(massive=massive), is_measurement=self.is_measurement)

    def sample(self,
               shots: int = 1024,
               massive: bool = False,
               reverse_endianness: bool = False) -> dict:
        """ Sample the state function as a normalized probability distribution. Returns dict of
        bitstrings in order of probability, with values being probability. """
        raise NotImplementedError
