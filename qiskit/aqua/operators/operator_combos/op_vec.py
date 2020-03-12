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

""" Eager Operator Vec Container """


import numpy as np
from functools import reduce

from qiskit.circuit import ParameterExpression

from .. import OperatorBase


class OpVec(OperatorBase):
    """ A class for storing and manipulating lists of operators. Vec here refers to the fact that this class serves
    as a base class for other Operator combinations which store a list of operators, such as OpSum or OpKron,
    but also refers to the "vec" mathematical operation.
    """

    def __init__(self, oplist, combo_fn=lambda x: x, coeff=1.0, param_bindings=None, abelian=False):
        """
        Args:
            oplist (list(OperatorBase)): The operators being summed.
            combo_fn (callable): The recombination function to reduce classical operators when available (e.g. sum)
            coeff (int, float, complex, ParameterExpression): A coefficient multiplying the primitive
            param_bindings(dict): A dictionary containing {param: list_of_bindings} mappings, such that each binding
            should be treated as a new op in oplist for that parameterization. Keys can also be ParameterVectors,
            or anything else that can be passed as a key in a Terra .bind_parameters call.

            Note that the default "recombination function" lambda above is the identity - it takes a list of operators,
            and is supposed to return a list of operators.
        """
        # Create copies of the oplist *pointers* for each binding. This should be very cheap. We can fix it if it's not.
        self._oplist = oplist
        self._combo_fn = combo_fn
        self._coeff = coeff
        self._param_bindings = param_bindings
        self._abelian = abelian

    @property
    def oplist(self):
        return self._oplist

    @property
    def combo_fn(self):
        return self._combo_fn

    @property
    def param_bindings(self):
        return self._param_bindings

    def num_parameterizations(self):
        return len(list(self._param_bindings.values())[0]) if self._param_bindings is not None else 1

    def get_parameterization(self, i):
        return {param: value_list[i] for (param, value_list) in self.param_bindings.items()}

    @property
    def abelian(self):
        return self._abelian

    # TODO: Keep this property for evals or just enact distribution at composition time?
    @property
    def distributive(self):
        """ Indicates whether the OpVec or subclass is distrubtive under composition. OpVec and OpSum are,
        meaning that opv @ op = opv[0] @ op + opv[1] @ op +... (plus for OpSum, vec for OpVec, etc.),
        while OpComposition and OpKron do not behave this way."""
        return True

    @property
    def coeff(self):
        return self._coeff

    def get_primitives(self):
        """ Return a set of strings describing the primitives contained in the Operator """
        return reduce(set.union, [op.get_primitives() for op in self.oplist])

    @property
    def num_qubits(self):
        """ For now, follow the convention that when one composes to a Vec, they are composing to each separate
        system. """
        # return sum([op.num_qubits for op in self.oplist])
        # TODO maybe do some check here that they're the same?
        return self.oplist[0].num_qubits

    # TODO change to *other to efficiently handle lists?
    def add(self, other):
        """ Addition. Overloaded by + in OperatorBase. OpSum overrides with its own add(). """
        if self == other:
            return self.mul(2.0)

        # TODO do this lazily for some primitives (Pauli, Instruction), and eager for others (Matrix)?
        # if eager and isinstance(other, OpPrimitive):
        #     return self.__class__([op.add(other) for op in self.oplist], coeff=self.coeff)

        # Avoid circular dependency
        from .op_sum import OpSum
        return OpSum([self, other])

    def neg(self):
        """ Negate. Overloaded by - in OperatorBase. """
        return self.mul(-1.0)

    def adjoint(self):
        """ Return operator adjoint (conjugate transpose). Overloaded by ~ in OperatorBase.

        Works for OpSum, OpCompose, OpVec, OpKron, at least. New combos must check whether they need to overload this.
        """
        # TODO test this a lot... probably different for OpKron.
        # TODO do this lazily? Basically rebuilds the entire tree, and ops and adjoints almost always come in pairs.
        return self.__class__([op.adjoint() for op in self.oplist], coeff=np.conj(self.coeff))

    def traverse(self, convert_fn, coeff=None):
        """ Apply the convert_fn to each node in the oplist. """
        return self.__class__([convert_fn(op) for op in self.oplist], coeff=coeff or self.coeff)

    def equals(self, other):
        """ Evaluate Equality. Overloaded by == in OperatorBase. """
        if not isinstance(other, type(self)) or not len(self.oplist) == len(other.oplist):
            return False
        # TODO test this a lot
        # Note, ordering matters here (i.e. different ordered lists will return False), maybe it shouldn't
        return self.oplist == other.oplist and self.param_bindings == other.param_bindings

    def mul(self, scalar):
        """ Scalar multiply. Overloaded by * in OperatorBase. """
        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))
        return self.__class__(self.oplist, coeff=self.coeff * scalar)

    def kron(self, other):
        """ Kron
        Note: You must be conscious of Qiskit's big-endian bit printing convention. Meaning, X.kron(Y)
        produces an X on qubit 0 and an Y on qubit 1, or X⨂Y, but would produce a QuantumCircuit which looks like
        -[Y]-
        -[X]-
        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.
        """
        # TODO do this lazily for some primitives (Matrix), and eager for others (Pauli, Instruction)?
        # NOTE: Doesn't work for OpComposition!
        # if eager and isinstance(other, OpPrimitive):
        #     return self.__class__([op.kron(other) for op in self.oplist], coeff=self.coeff)

        # Avoid circular dependency
        from .op_kron import OpKron
        return OpKron([self, other])

    def kronpower(self, other):
        """ Kron with Self Multiple Times """
        # Hack to make op1^(op2^0) work as intended.
        if other == 0:
            return 1
        if not isinstance(other, int) or other <= 0:
            raise TypeError('Kronpower can only take positive int arguments')

        # Avoid circular dependency
        from .op_kron import OpKron
        return OpKron([self]*other)

    # TODO change to *other to efficiently handle lists?
    def compose(self, other):
        """ Operator Composition (Linear algebra-style, right-to-left)

        Note: You must be conscious of Quantum Circuit vs. Linear Algebra ordering conventions. Meaning, X.compose(Y)
        produces an X∘Y on qubit 0, but would produce a QuantumCircuit which looks like
        -[Y]-[X]-
        Because Terra prints circuits with the initial state at the left side of the circuit.
        """

        # TODO do this lazily for some primitives (Matrix), and eager for others (Pauli, Instruction)?
        # if eager and isinstance(other, OpPrimitive):
        #     return self.__class__([op.compose(other) for op in self.oplist], coeff=self.coeff)

        # Avoid circular dependency
        from .op_composition import OpComposition
        return OpComposition([self, other])

    def power(self, other):
        """ Compose with Self Multiple Times """
        if not isinstance(other, int) or other <= 0:
            raise TypeError('power can only take positive int arguments')

        # Avoid circular dependency
        from .op_composition import OpComposition
        return OpComposition([self]*other)

    def to_matrix(self, massive=False):
        """ Return numpy matrix of operator, warn if more than 16 qubits to force the user to set massive=True if
        they want such a large matrix. Generally big methods like this should require the use of a converter,
        but in this case a convenience method for quick hacking and access to classical tools is appropriate. """

        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError('to_matrix will return an exponentially large matrix, in this case {0}x{0} elements.'
                             ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        # Combination function must be able to handle classical values
        # TODO wrap combo function in np.array? Or just here to make sure broadcasting works?
        if self.distributive:
            return self.combo_fn([op.to_matrix()*self.coeff for op in self.oplist])
        else:
            return self.combo_fn([op.to_matrix() for op in self.oplist]) * self.coeff

    def eval(self, front=None, back=None):
        """ A square binary Operator can be defined as a function over two binary strings of equal length. This
        method returns the value of that function for a given pair of binary strings. For more information,
        see the eval method in operator_base.py.

        OpVec's eval recursively evaluates each Operator in self.oplist's eval, and returns a value based on the
        recombination function.


        # TODO this doesn't work for compositions and krons! Needs to be to_matrix.
        """
        # The below code only works for distributive OpVecs, e.g. OpVec and OpSum
        if not self.distributive:
            return NotImplementedError

        # TODO Do we need to use partial(np.sum, axis=0) as OpSum combo to be able to handle vector returns correctly?
        if isinstance(front, list):
            return [self.eval(front_elem, back=back) for front_elem in front]

        from ..state_functions import StateFn

        if back is not None and not isinstance(back, OperatorBase):
            back = StateFn(back, is_measurement=True)

        res = []
        for op in self.oplist:
            if isinstance(op, StateFn):
                new_front = (self.coeff * op).eval(front)
                res += [back.eval(new_front)] if back is not None else [new_front]
            else:
                res += [(self.coeff*op).eval(front, back)]

        return self.combo_fn(res)

    def exp_i(self):
        """ Raise Operator to power e ^ (i * op)"""
        from qiskit.aqua.operators import OpEvolution
        return OpEvolution(self)

    def __str__(self):
        """Overload str() """
        main_string = "{}([{}])".format(self.__class__.__name__, ', '.join([str(op) for op in self.oplist]))
        if self.abelian:
            main_string = 'Abelian' + main_string
        if not self.coeff == 1.0:
            main_string = '{} * '.format(self.coeff) + main_string
        return main_string

    def __repr__(self):
        """Overload str() """
        return "{}({}, coeff={}, abelian={})".format(self.__class__.__name__,
                                                     repr(self.oplist),
                                                     self.coeff,
                                                     self.abelian)

    def bind_parameters(self, param_dict):
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            if self.coeff in unrolled_dict:
                # TODO what do we do about complex?
                param_value = float(self.coeff.bind(unrolled_dict[self.coeff]))
        return self.traverse(lambda x: x.bind_parameters(param_dict), coeff=param_value)

    def print_details(self):
        """ print details """
        raise NotImplementedError

    def reduce(self):
        reduced_ops = [op.reduce() for op in self.oplist]
        return self.__class__(reduced_ops, coeff=self.coeff)
