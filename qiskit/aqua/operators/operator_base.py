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

""" Base Operator """

from abc import ABC, abstractmethod
import numpy as np

from qiskit.circuit import ParameterExpression, ParameterVector

from qiskit.aqua import AquaError


class OperatorBase(ABC):
    """ An square binary Operator can be defined in a two equivalent ways:
        1) A functional, taking a complex function over a binary alphabet
           of values to another binary function
        2) A complex function over two values of a binary alphabet

    """

    @property
    def name(self):
        """ returns name """
        return self._name

    @name.setter
    def name(self, new_value):
        """ sets name """
        self._name = new_value

    # TODO replace with proper alphabets later?
    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """ returns number of qubits """
        raise NotImplementedError

    @abstractmethod
    def get_primitives(self) -> set:
        """ Return a set of strings describing the primitives contained in the Operator """
        raise NotImplementedError

    # TODO allow massive argument to decide whether to perform to_matrix?
    @abstractmethod
    def eval(self, front=None):
        """
        # TODO update
        A square binary Operator can be defined as a function over two binary strings
        of equal length,
        or equivalently, a function taking a binary function to another binary function.
        This method returns the
        value of that function for a given pair of binary strings if both front and back are
        supplied, or returns a
        StateFn if only front is provided. Note that providing both values is simply a shorthand for
        op.eval(front).eval(back) if back is a binary string, or back.eval(op.eval(front)) if back
        is a Measurement (front can be a StateFn or binary string in either case).

        A brute force way to evaluate an expectation for some **positive
        real** state function sf would be:
            sampled_strings = sf.sample(shots=1000)
            sum([op.eval(bstr, bstr) for bstr in sampled_strings])
        or, exactly:
            sum([op.eval(bstr, bstr) * sf.eval(bstr) for bstr in sampled_strings])

        However, for a quantum state function, i.e. a complex state function, this would need to be:
            sampled_strings = sf.sample(shots=1000)
            sum([op.eval(bstr, bstr) * np.sign(sf.eval(bstr)) for bstr in sampled_strings])
        or, exactly:
            sum([op.eval(bstr, bstr) * np.conj(sf.eval(bstr)) * sf.eval(bstr)
            for bstr in sampled_strings])

        Note that for a quantum state function, we do not generally
        have efficient classical access to sf.sample or
        sf.eval.

        """
        raise NotImplementedError

    @abstractmethod
    def reduce(self):
        """ Try collapsing the Operator structure, usually after some
        time of processing. E.g. a conversion,
        some operators in an ComposedOp can now be directly composed.
        At worst, just returns self."""
        raise NotImplementedError

    @abstractmethod
    def to_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return numpy vector representing StateFn evaluated on each basis state. Warn if more
        than 16 qubits to force having to set massive=True if such a large vector is desired.
        Generally a conversion method like this may require the use of a converter,
        but in this case a convenience method for quick hacking and access to
        classical tools is appropriate. """
        raise NotImplementedError

# Addition / Subtraction

    def __add__(self, other):
        """ Overload + operation """
        # Hack to be able to use sum(list_of_ops) nicely,
        # because sum adds 0 to the first element of the list.
        if other == 0:
            return self

        return self.add(other)

    def __radd__(self, other):
        """ Overload right + operation """
        # Hack to be able to use sum(list_of_ops) nicely,
        # because sum adds 0 to the first element of the list.
        if other == 0:
            return self

        return self.add(other)

    @abstractmethod
    def add(self, other):
        """ Addition """
        raise NotImplementedError

    def __sub__(self, other):
        """ Overload + operation """
        return self.add(-other)

    def __rsub__(self, other):
        """ Overload right + operation """
        return self.neg().add(other)

# Negation

    def __neg__(self):
        """ Overload unary - """
        return self.neg()

    @abstractmethod
    def neg(self):
        """ Return operator negation """
        raise NotImplementedError

# Adjoint

    def __invert__(self):
        """ Overload unary ~ """
        return self.adjoint()

    @abstractmethod
    def adjoint(self):
        """ Return operator adjoint """
        raise NotImplementedError

    # Equality

    def __eq__(self, other):
        """ Overload == operation """
        return self.equals(other)

    @abstractmethod
    def equals(self, other):
        """ Evaluate Equality """
        raise NotImplementedError

# Scalar Multiplication

    def __mul__(self, other):
        """ Overload * """
        return self.mul(other)

    def __rmul__(self, other):
        """ Overload * """
        return self.mul(other)

    def __truediv__(self, other):
        """ Overload / """
        return self.mul(1 / other)

    @abstractmethod
    def mul(self, scalar):
        """ Scalar multiply """
        raise NotImplementedError

    def __xor__(self, other):
        """ Overload ^ for tensor or tensorpower if ^ is int"""
        if isinstance(other, int):
            return self.tensorpower(other)
        else:
            return self.tensor(other)

    # Hack to make (I^0)^Z work as intended.
    def __rxor__(self, other):
        """ Overload ^ for tensor or tensorpower if ^ is int"""
        if other == 1:
            return self
        else:
            return other.tensor(self)

    # Copy from terra, except the list unrolling:
    @staticmethod
    def _unroll_param_dict(value_dict):
        unrolled_value_dict = {}
        for (param, value) in value_dict.items():
            if isinstance(param, ParameterExpression):
                unrolled_value_dict[param] = value
            if isinstance(param, ParameterVector):
                if not len(param) == len(value):
                    raise ValueError(
                        'ParameterVector {} has length {}, which differs from value list {} of '
                        'len {}'.format(param, len(param), value, len(value)))
                unrolled_value_dict.update(zip(param, value))
        if isinstance(list(unrolled_value_dict.values())[0], list):
            # check that all are same length
            unrolled_value_dict_list = []
            try:
                for i in range(len(list(unrolled_value_dict.values())[0])):
                    unrolled_value_dict_list.append(
                        OperatorBase._get_param_dict_for_index(unrolled_value_dict, i))
                return unrolled_value_dict_list
            except IndexError:
                raise AquaError('Parameter binding lists must all be the same length.')
        return unrolled_value_dict

    @staticmethod
    def _get_param_dict_for_index(unrolled_dict, i):
        return {k: v[i] for (k, v) in unrolled_dict.items()}

    @abstractmethod
    def tensor(self, other):
        """ Tensor product """
        raise NotImplementedError

    @abstractmethod
    def tensorpower(self, other):
        """ Tensor product with Self Multiple Times """
        raise NotImplementedError

# Composition

    def __matmul__(self, other):
        """ Overload @ for composition"""
        return self.compose(other)

    @abstractmethod
    def compose(self, other):
        """ Operator Composition (Linear Algebra-style, right-to-left) """
        raise NotImplementedError

    @abstractmethod
    def power(self, other):
        """ Compose with Self Multiple Times """
        raise NotImplementedError

    def __pow__(self, other):
        """ Overload ** for power"""
        return self.power(other)

# Printing

    @abstractmethod
    def __str__(self) -> str:
        """Overload str() """
        raise NotImplementedError
