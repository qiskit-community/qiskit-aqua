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

""" ListOp Operator Class """

from typing import List, Union, Optional, Callable, Iterator, Set
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
        """ Returns the list of ``OperatorBase`` defining the underlying function of this
        Operator.  """
        return self._oplist

    @property
    def combo_fn(self) -> Callable:
        """ The function defining how to combine ``oplist`` (or Numbers, or NumPy arrays) to
        produce the Operator's underlying function. For example, SummedOp's combination function
        is to add all of the Operators in ``oplist``. """
        return self._combo_fn

    @property
    def abelian(self) -> bool:
        """ Whether the Operators in ``OpList`` are known to commute with one another. """
        return self._abelian

    # TODO: Keep this property for evals or just enact distribution at composition time?
    @property
    def distributive(self) -> bool:
        """ Indicates whether the ListOp or subclass is distributive under composition.
        ListOp and SummedOp are, meaning that (opv @ op) = (opv[0] @ op + opv[1] @ op)
        (using plus for SummedOp, list for ListOp, etc.), while ComposedOp and TensoredOp
        do not behave this way."""
        return True

    @property
    def coeff(self) -> Union[int, float, complex, ParameterExpression]:
        """ The scalar coefficient multiplying the Operator. """
        return self._coeff

    def primitive_strings(self) -> Set[str]:
        return reduce(set.union, [op.primitive_strings() for op in self.oplist])

    @property
    def num_qubits(self) -> int:
        """ The number of qubits over which the Operator is defined. For now, follow the
        convention that when one composes to a ListOp, they are composing to each separate
        system. """
        return self.oplist[0].num_qubits

    def add(self, other: OperatorBase) -> OperatorBase:
        if self == other:
            return self.mul(2.0)

        # Avoid circular dependency
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .summed_op import SummedOp
        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        # TODO do this lazily? Basically rebuilds the entire tree, and ops and adjoints almost
        #  always come in pairs, so an AdjointOp holding a reference could save copying.
        return self.__class__([op.adjoint() for op in self.oplist], coeff=np.conj(self.coeff))

    def traverse(self,
                 convert_fn: Callable,
                 coeff: Optional[Union[int, float, complex,
                                       ParameterExpression]] = None) -> OperatorBase:
        """ Apply the convert_fn to each node in the oplist. """
        return self.__class__([convert_fn(op) for op in self.oplist], coeff=coeff or self.coeff)

    def equals(self, other: OperatorBase) -> bool:
        if not isinstance(other, type(self)) or not len(self.oplist) == len(other.oplist):
            return False
        # Note, ordering matters here (i.e. different list orders will return False)
        return all([op1 == op2 for op1, op2 in zip(self.oplist, other.oplist)])

    # We need to do this because otherwise Numpy takes over scalar multiplication and wrecks it if
    # isinstance(scalar, np.number) - this started happening when we added __get_item__().
    __array_priority__ = 10000

    def mul(self, scalar: Union[int, float, complex, ParameterExpression]) -> OperatorBase:
        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))
        return self.__class__(self.oplist, coeff=self.coeff * scalar)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        """ Return tensor product between self and other, overloaded by ``^``.
        Note: You must be conscious of Qiskit's big-endian bit printing convention.
        Meaning, X.tensor(Y) produces an X on qubit 0 and an Y on qubit 1, or X⨂Y, but would
        produce a QuantumCircuit which looks like
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
        r"""
        Return Operator Composition between self and other (linear algebra-style:
        A@B(x) = A(B( x))), overloaded by ``@``.

        Note: You must be conscious of Quantum Circuit vs. Linear Algebra ordering conventions.
        Meaning, X.compose(Y) produces an X∘Y on qubit 0, but would produce a QuantumCircuit
        which looks like
            -[Y]-[X]-
        because Terra prints circuits with the initial state at the left side of the circuit.
        """
        # Avoid circular dependency
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .composed_op import ComposedOp
        return ComposedOp([self, other])

    def power(self, exponent: int) -> OperatorBase:
        if not isinstance(exponent, int) or exponent <= 0:
            raise TypeError('power can only take positive int arguments')

        # Avoid circular dependency
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .composed_op import ComposedOp
        return ComposedOp([self] * exponent)

    def to_matrix(self, massive: bool = False) -> np.ndarray:
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
        """ Returns SciPy sparse matrix representation of the Operator. """

        # Combination function must be able to handle classical values
        if self.distributive:
            return self.combo_fn([op.to_spmatrix() * self.coeff for op in self.oplist])
        else:
            return self.combo_fn([op.to_spmatrix() for op in self.oplist]) * self.coeff

    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        """
        Evaluate the Operator's underlying function, either on a binary string or another Operator.
        See the eval method in operator_base.py.

        ListOp's eval recursively evaluates each Operator in ``oplist``,
        and combines the results using the recombination function ``combo_fn``.
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
        """ Raise Operator to power e ^ (-i * op)"""
        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.operators import EvolvedOp
        return EvolvedOp(self)

    def __str__(self) -> str:
        main_string = "{}(\n[{}])".format(self.__class__.__name__, ',\n'.join(
            [str(op) for op in self.oplist]))
        if self.abelian:
            main_string = 'Abelian' + main_string
        if self.coeff != 1.0:
            main_string = '{} * '.format(self.coeff) + main_string
        return main_string

    def __repr__(self) -> str:
        return "{}({}, coeff={}, abelian={})".format(self.__class__.__name__,
                                                     repr(self.oplist),
                                                     self.coeff,
                                                     self.abelian)

    def bind_parameters(self, param_dict: dict) -> OperatorBase:
        """ Bind parameter values to ``ParameterExpressions`` in ``coeff`` or ``primitive``. """
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
        """ Returns an equivalent Operator composed of only NumPy-based primitives, such as
        ``MatrixOp`` and ``VectorStateFn``. """
        return self.__class__([op.to_matrix_op(massive=massive) for op in self.oplist],
                              coeff=self.coeff).reduce()

    def to_circuit_op(self) -> OperatorBase:
        """ Returns an equivalent Operator composed of only QuantumCircuit-based primitives,
        such as ``CircuitOp`` and ``CircuitStateFn``. """
        return self.__class__([op.to_circuit_op() for op in self.oplist],
                              coeff=self.coeff).reduce()

    def to_pauli_op(self, massive: bool = False) -> OperatorBase:
        """ Returns an equivalent Operator composed of only Pauli-based primitives,
        such as ``PauliOp``. """
        # pylint: disable=cyclic-import
        from ..state_functions.state_fn import StateFn
        return self.__class__([op.to_pauli_op(massive=massive)
                               if not isinstance(op, StateFn) else op
                               for op in self.oplist], coeff=self.coeff).reduce()

    # Array operations:

    def __getitem__(self, offset: int) -> OperatorBase:
        return self.oplist[offset]

    def __iter__(self) -> Iterator:
        return iter(self.oplist)

    def __len__(self) -> int:
        return len(self.oplist)
