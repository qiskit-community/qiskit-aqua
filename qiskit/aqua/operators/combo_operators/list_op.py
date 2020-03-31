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

""" Eager Operator Vec Container """

from typing import List, Union, Optional, Callable, Iterator
from functools import reduce
import numpy as np
from scipy.sparse import spmatrix

from qiskit.circuit import ParameterExpression

from ..operator_base import OperatorBase


class ListOp(OperatorBase):
    """ A class for storing and manipulating lists of operators.
    Vec here refers to the fact that this class serves
    as a base class for other Operator combinations which store
    a list of operators, such as SummedOp or TensoredOp,
    but also refers to the "vec" mathematical operation.
    """

    def __init__(self,
                 oplist: List[OperatorBase],
                 combo_fn: Callable = lambda x: x,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 abelian: bool = False) -> None:
        """
        Args:
            oplist: The operators being summed.
            combo_fn (callable): The recombination function to reduce classical operators
            when available (e.g. sum)
            coeff: A coefficient multiplying the operator
            abelian: indicates if abelian

            Note that the default "recombination function" lambda above is the identity -
            it takes a list of operators,
            and is supposed to return a list of operators.
        """
        self._oplist = oplist
        self._combo_fn = combo_fn
        self._coeff = coeff
        self._abelian = abelian

    @property
    def oplist(self) -> List[OperatorBase]:
        """ returns op list """
        return self._oplist

    @property
    def combo_fn(self) -> Callable:
        """ returns combo function """
        return self._combo_fn

    @property
    def abelian(self) -> bool:
        """ returns abelian """
        return self._abelian

    # TODO: Keep this property for evals or just enact distribution at composition time?
    @property
    def distributive(self) -> bool:
        """ Indicates whether the ListOp or subclass is distributive under composition.
        ListOp and SummedOp are,
        meaning that opv @ op = opv[0] @ op + opv[1] @ op +... (plus for SummedOp,
        vec for ListOp, etc.),
        while ComposedOp and TensoredOp do not behave this way."""
        return True

    @property
    def coeff(self) -> Union[int, float, complex, ParameterExpression]:
        """ returns coeff """
        return self._coeff

    def get_primitives(self) -> set:
        """ Return a set of strings describing the primitives contained in the Operator """
        return reduce(set.union, [op.get_primitives() for op in self.oplist])

    @property
    def num_qubits(self) -> int:
        """ For now, follow the convention that when one composes to a Vec,
        they are composing to each separate system. """
        # return sum([op.num_qubits for op in self.oplist])
        # TODO maybe do some check here that they're the same?
        return self.oplist[0].num_qubits

    # TODO change to *other to efficiently handle lists?
    def add(self, other: OperatorBase) -> OperatorBase:
        """ Addition. Overloaded by + in OperatorBase. SummedOp overrides with its own add(). """
        if self == other:
            return self.mul(2.0)

        # TODO do this lazily for some primitives (Pauli, Instruction),
        # and eager for others (Matrix)?
        # if eager and isinstance(other, PrimitiveOp):
        #     return self.__class__([op.add(other) for op in self.oplist], coeff=self.coeff)

        # Avoid circular dependency
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .summed_op import SummedOp
        return SummedOp([self, other])

    def neg(self) -> OperatorBase:
        """ Negate. Overloaded by - in OperatorBase. """
        return self.mul(-1.0)

    def adjoint(self) -> OperatorBase:
        """ Return operator adjoint (conjugate transpose). Overloaded by ~ in OperatorBase.

        Works for SummedOp, ComposedOp, ListOp, TensoredOp, at least.
        New combos must check whether they need to overload this.
        """
        # TODO test this a lot... probably different for TensoredOp.
        # TODO do this lazily? Basically rebuilds the entire tree,
        #  and ops and adjoints almost always come in pairs.
        return self.__class__([op.adjoint() for op in self.oplist], coeff=np.conj(self.coeff))

    def traverse(self,
                 convert_fn: Callable,
                 coeff: Optional[Union[int, float, complex,
                                       ParameterExpression]] = None) -> OperatorBase:
        """ Apply the convert_fn to each node in the oplist. """
        return self.__class__([convert_fn(op) for op in self.oplist], coeff=coeff or self.coeff)

    def equals(self, other: OperatorBase) -> bool:
        """ Evaluate Equality. Overloaded by == in OperatorBase. """
        if not isinstance(other, type(self)) or not len(self.oplist) == len(other.oplist):
            return False
        # TODO test this a lot
        # Note, ordering matters here (i.e. different ordered lists
        # will return False), maybe it shouldn't
        return self.oplist == other.oplist

    # We need to do this because otherwise Numpy takes over scalar multiplication and wrecks it if
    # isinstance(scalar, np.number) - this started happening when we added __get_item__().
    __array_priority__ = 10000

    def mul(self, scalar: Union[int, float, complex, ParameterExpression]) -> OperatorBase:
        """ Scalar multiply. Overloaded by * in OperatorBase. """
        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))
        return self.__class__(self.oplist, coeff=self.coeff * scalar)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        """ Tensor product
        Note: You must be conscious of Qiskit's big-endian bit printing convention.
        Meaning, X.tensor(Y)
        produces an X on qubit 0 and an Y on qubit 1, or X⨂Y, but would produce a
        QuantumCircuit which looks like
        -[Y]-
        -[X]-
        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.
        """
        # TODO do this lazily for some primitives (Matrix), and eager
        #  for others (Pauli, Instruction)?
        # NOTE: Doesn't work for ComposedOp!
        # if eager and isinstance(other, PrimitiveOp):
        #     return self.__class__([op.tensor(other) for op in self.oplist], coeff=self.coeff)

        # Avoid circular dependency
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .tensored_op import TensoredOp
        return TensoredOp([self, other])

    def tensorpower(self, other: int) -> Union[OperatorBase, int]:
        """ Tensor product with Self Multiple Times """
        # Hack to make op1^(op2^0) work as intended.
        if other == 0:
            return 1
        if not isinstance(other, int) or other <= 0:
            raise TypeError('Tensorpower can only take positive int arguments')

        # Avoid circular dependency
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .tensored_op import TensoredOp
        return TensoredOp([self] * other)

    # TODO change to *other to efficiently handle lists?
    def compose(self, other: OperatorBase) -> OperatorBase:
        """ Operator Composition (Linear algebra-style, right-to-left)

        Note: You must be conscious of Quantum Circuit vs. Linear Algebra ordering conventions.
        Meaning, X.compose(Y)
        produces an X∘Y on qubit 0, but would produce a QuantumCircuit which looks like
        -[Y]-[X]-
        Because Terra prints circuits with the initial state at the left side of the circuit.
        """

        # TODO do this lazily for some primitives (Matrix), and eager
        #  for others (Pauli, Instruction)?
        # if eager and isinstance(other, PrimitiveOp):
        #     return self.__class__([op.compose(other) for op in self.oplist], coeff=self.coeff)

        # Avoid circular dependency
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .composed_op import ComposedOp
        return ComposedOp([self, other])

    def power(self, other: int) -> OperatorBase:
        """ Compose with Self Multiple Times """
        if not isinstance(other, int) or other <= 0:
            raise TypeError('power can only take positive int arguments')

        # Avoid circular dependency
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .composed_op import ComposedOp
        return ComposedOp([self] * other)

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return numpy vector representing StateFn evaluated on each basis state. Warn if more
        than 16 qubits to force having to set massive=True if such a large vector is desired.
        Generally a conversion method like this may require the use of a converter,
        but in this case a convenience method for quick hacking and access to
        classical tools is appropriate. """

        if self.num_qubits > 16 and not massive:
            # TODO figure out sparse matrices?
            raise ValueError(
                'to_matrix will return an exponentially large matrix, '
                'in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        # Combination function must be able to handle classical values
        # TODO wrap combo function in np.array? Or just here to make sure broadcasting works?
        if self.distributive:
            return self.combo_fn([op.to_matrix() * self.coeff for op in self.oplist])
        else:
            return self.combo_fn([op.to_matrix() for op in self.oplist]) * self.coeff

    def to_spmatrix(self) -> spmatrix:
        """ Return numpy matrix of operator, warn if more than 16 qubits
        to force the user to set massive=True if
        they want such a large matrix. Generally big methods like this should
        require the use of a converter,
        but in this case a convenience method for quick hacking and access to
        classical tools is appropriate. """

        # Combination function must be able to handle classical values
        if self.distributive:
            return self.combo_fn([op.to_spmatrix() * self.coeff for op in self.oplist])
        else:
            return self.combo_fn([op.to_spmatrix() for op in self.oplist]) * self.coeff

    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        """ A square binary Operator can be defined as a function over two binary strings
        of equal length. This
        method returns the value of that function for a given pair of binary strings.
        For more information,
        see the eval method in operator_base.py.

        ListOp's eval recursively evaluates each Operator in self.oplist's eval,
        and returns a value based on the
        recombination function.


        # TODO this doesn't work for compositions and tensors! Needs to be to_matrix.
        """
        # The below code only works for distributive ListOps, e.g. ListOp and SummedOp
        if not self.distributive:
            return NotImplementedError

        evals = [(self.coeff * op).eval(front) for op in self.oplist]
        if all([isinstance(op, OperatorBase) for op in evals]):
            return self.__class__(evals)
        elif any([isinstance(op, OperatorBase) for op in evals]):
            raise TypeError('Cannot handle mixed scalar and Operator eval results.')
        else:
            return self.combo_fn(evals)

    def exp_i(self) -> OperatorBase:
        """ Raise Operator to power e ^ (i * op)"""
        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.operators import EvolvedOp
        return EvolvedOp(self)

    def __str__(self) -> str:
        """Overload str() """
        main_string = "{}(\n[{}])".format(self.__class__.__name__, ',\n'.join(
            [str(op) for op in self.oplist]))
        if self.abelian:
            main_string = 'Abelian' + main_string
        if self.coeff != 1.0:
            main_string = '{} * '.format(self.coeff) + main_string
        return main_string

    def __repr__(self) -> str:
        """Overload str() """
        return "{}({}, coeff={}, abelian={})".format(self.__class__.__name__,
                                                     repr(self.oplist),
                                                     self.coeff,
                                                     self.abelian)

    def bind_parameters(self, param_dict: dict) -> OperatorBase:
        """ bind parameters """
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                return ListOp([self.bind_parameters(param_dict) for param_dict in unrolled_dict])
            coeff_param = list(self.coeff.parameters)[0]
            if coeff_param in unrolled_dict:
                # TODO what do we do about complex?
                value = unrolled_dict[coeff_param]
                param_value = float(self.coeff.bind({coeff_param: value}))
        return self.traverse(lambda x: x.bind_parameters(param_dict), coeff=param_value)

    def reduce(self) -> OperatorBase:
        reduced_ops = [op.reduce() for op in self.oplist]
        return self.__class__(reduced_ops, coeff=self.coeff)

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """ Return a MatrixOp for this operator. """
        return self.__class__([op.to_matrix_op(massive=massive) for op in self.oplist]).reduce()

    # Array operations:

    def __getitem__(self, offset: int) -> OperatorBase:
        return self.oplist[offset]

    def __iter__(self) -> Iterator:
        return iter(self.oplist)

    def __len__(self) -> int:
        return len(self.oplist)
