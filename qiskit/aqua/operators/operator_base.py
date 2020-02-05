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

""" Base Operator """

from abc import ABC, abstractmethod


class OperatorBase(ABC):
    """ An square binary Operator can be defined in a two equivalent ways:
        1) A functional, taking a complex function over a binary alphabet of values to another binary function
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
    @abstractmethod
    def num_qubits(self):
        raise NotImplementedError

    @abstractmethod
    def get_primitives(self):
        """ Return a set of primitives in the Operator """
        raise NotImplementedError

    @abstractmethod
    def eval(self, val1, val2):
        """ A square binary Operator can be defined as a function over two binary strings of equal length. This
        method returns the value of that function for a given pair of binary strings. Note that by using
        functools.partial, a function over a single binary string is returned, which is equivalent to a state
        function.

        A brute force way to evaluate an expectation for some **positive real** state function sf would be:
            sampled_strings = sf.sample(shots=1000)
            sum([op.eval(bstr, bstr) for bstr in sampled_strings])
        or, exactly:
            sum([op.eval(bstr, bstr) * sf.eval(bstr) for bstr in sampled_strings])

        However, for a quantum state function, i.e. a complex state function, this would need to be:
            sampled_strings = sf.sample(shots=1000)
            sum([op.eval(bstr, bstr) * np.sign(sf.eval(bstr)) for bstr in sampled_strings])
        or, exactly:
            sum([op.eval(bstr, bstr) * np.conj(sf.eval(bstr)) * sf.eval(bstr) for bstr in sampled_strings])

        Note that for a quantum statefunction, we do not generally have efficient classical access to sf.sample or
        sf.eval.

        """
        raise NotImplementedError

# Addition / Subtraction

    def __add__(self, other):
        """ Overload + operation """
        return self.add(other)

    def __radd__(self, other):
        """ Overload right + operation """
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

    @abstractmethod
    def mul(self, scalar):
        """ Scalar multiply """
        raise NotImplementedError

    def __xor__(self, other):
        """ Overload ^ for kron or kronpower if ^ is int"""
        if isinstance(other, int):
            return self.kronpower(other)
        else:
            return self.kron(other)

    @abstractmethod
    def kron(self, other):
        """ Kron """
        raise NotImplementedError

    # TODO add lazy option?
    @abstractmethod
    def kronpower(self, other):
        """ Kron with Self Multiple Times """
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
    def __str__(self):
        """Overload str() """
        raise NotImplementedError

    @abstractmethod
    def print_details(self):
        """ print details """
        raise NotImplementedError
