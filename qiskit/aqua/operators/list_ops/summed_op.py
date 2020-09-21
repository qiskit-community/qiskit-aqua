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

""" SummedOp Class """

from typing import List, Union, cast, Callable
import warnings

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from .list_op import ListOp
from ..legacy.base_operator import LegacyBaseOperator
from ..legacy.weighted_pauli_operator import WeightedPauliOperator
from ..operator_base import OperatorBase
from ... import AquaError


class SummedOp(ListOp):
    """ A class for lazily representing sums of Operators. Often Operators cannot be
    efficiently added to one another, but may be manipulated further so that they can be
    later. This class holds logic to indicate that the Operators in ``oplist`` are meant to
    be added together, and therefore if they reach a point in which they can be, such as after
    evaluation or conversion to matrices, they can be reduced by addition. """

    def __init__(self,
                 oplist: List[OperatorBase],
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 abelian: bool = False) -> None:
        """
        Args:
            oplist: The Operators being summed.
            coeff: A coefficient multiplying the operator
            abelian: Indicates whether the Operators in ``oplist`` are known to mutually commute.
        """
        super().__init__(oplist,
                         combo_fn=lambda x: np.sum(x, axis=0),
                         coeff=coeff,
                         abelian=abelian)

    @property
    def num_qubits(self) -> int:
        return self.oplist[0].num_qubits

    @property
    def distributive(self) -> bool:
        return True

    def add(self, other: OperatorBase) -> OperatorBase:
        """Return Operator addition of ``self`` and ``other``, overloaded by ``+``.

        Note:
            This appends ``other`` to ``self.oplist`` without checking ``other`` is already
            included or not. If you want to simplify them, please use :meth:`simplify`.

        Args:
            other: An ``OperatorBase`` with the same number of qubits as self, and in the same
                'Operator', 'State function', or 'Measurement' category as self (i.e. the same type
                of underlying function).

        Returns:
            A ``SummedOp`` equivalent to the sum of self and other.
        """
        self_new_ops = self.oplist if self.coeff == 1 \
            else [op.mul(self.coeff) for op in self.oplist]
        if isinstance(other, SummedOp):
            other_new_ops = other.oplist if other.coeff == 1 \
                else [op.mul(other.coeff) for op in other.oplist]
        else:
            other_new_ops = [other]
        return SummedOp(self_new_ops + other_new_ops)

    def collapse_summands(self) -> 'SummedOp':
        """Return Operator by simplifying duplicate operators.

        E.g., ``SummedOp([2 * X ^ Y, X ^ Y]).collapse_summands() -> SummedOp([3 * X ^ Y])``.

        Returns:
            A simplified ``SummedOp`` equivalent to self.
        """
        from qiskit.aqua.operators import PrimitiveOp
        oplist = []  # type: List[OperatorBase]
        coeffs = []  # type: List[Union[int, float, complex, ParameterExpression]]
        for op in self.oplist:
            if isinstance(op, PrimitiveOp):
                new_op = PrimitiveOp(op.primitive)
                new_coeff = op.coeff * self.coeff
                if new_op in oplist:
                    index = oplist.index(new_op)
                    coeffs[index] += new_coeff
                else:
                    oplist.append(new_op)
                    coeffs.append(new_coeff)
            else:
                if op in oplist:
                    index = oplist.index(op)
                    coeffs[index] += self.coeff
                else:
                    oplist.append(op)
                    coeffs.append(self.coeff)
        new_summed_op = SummedOp([op * coeff for op, coeff in zip(oplist, coeffs)])  # type: ignore
        return new_summed_op.filter_ops_by_coeff(0, 0)

    def filter_ops_by_coeff(self,
                            lower_bound: Union[int, float, complex],
                            upper_bound: Union[int, float, complex]) -> 'SummedOp':
        """
        Create a new SummedOp that only has summands with coefficients that lie outside of
        lower_bound' and 'upper_bound'. Be wary of filtering across ranges including 1 as
        it's the default coefficient for subclasses of OperatorBase.

        For complex coefficients and bounds a + j * b <= x + j * y if a <= b and x <= y.
        This is a partial order because some complex numbers are incomparable
        (e.g. 1 - j and - 1 + j). Summands with incomparable complex coefficients are not
        filtered out. See in-line comments for details.

        Args:
            lower_bound: The lower bound of the filter.
            upper_bound: The upper bound of the filter.

        Returns:
            A new ListOp that contains only the operators with coefficients outside of the given
            bounds. If all summands are filtered out, returns a singleton sum of 0 * I with
            self.coeff = 1.0.
        """

        def within_bound(coefficient: Union[int, float, complex, ParameterExpression],
                         bound: Union[int, float, complex],
                         comparator: Callable[[Union[int, float],
                                               Union[int, float]], bool]) -> bool:
            if isinstance(coefficient, complex):
                within_real_bound = comparator(coefficient.real, bound.real)
                if isinstance(bound, complex):
                    within_imag_bound = comparator(coefficient.imag, bound.imag)
                    if within_real_bound and within_imag_bound:
                        return True  # comparable and inside bounds, strip coefficient
                    elif within_real_bound ^ within_imag_bound:
                        return False  # incomparable, don't strip coefficient
                    else:
                        return False  # comparable and outside bounds, don't strip coefficient
                else:
                    return within_real_bound  # filter only real part of coefficient
            elif isinstance(coefficient, ParameterExpression):
                return True
            else:
                return comparator(coefficient, bound.real)

        def within_bounds(coefficient: Union[int, float, complex, ParameterExpression]) -> bool:
            return within_bound(coefficient, lower_bound, lambda x, y: x >= y) \
                   and within_bound(coefficient, upper_bound, lambda x, y: x <= y)

        from qiskit.aqua.operators import I, PrimitiveOp, StateFn
        new_op_list = []  # type: List[OperatorBase]
        if not within_bounds(self.coeff):
            for op in self.oplist:
                if isinstance(op, SummedOp):
                    if not within_bounds(op.coeff):
                        new_op = op.filter_ops_by_coeff(lower_bound, upper_bound)
                        new_op_list.append(new_op)
                elif isinstance(op, (StateFn, PrimitiveOp)):
                    if not within_bounds(op.coeff):
                        new_op_list.append(op)
                else:
                    new_op_list.append(op)
            return SummedOp(new_op_list, self.coeff, self.abelian)
        else:
            return SummedOp([0.0 * I.tensorpower(self.num_qubits)], 1.0, self.abelian)

    # TODO be smarter about the fact that any two ops in oplist could be evaluated for sum.
    def reduce(self) -> OperatorBase:
        """Try collapsing list or trees of sums.

        Tries to sum up duplicate operators and reduces the operators
        in the sum.

        Returns:
            A collapsed version of self, if possible.
        """
        # reduce constituents
        reduced_ops = sum(op.reduce() for op in self.oplist) * self.coeff

        # group duplicate operators
        if isinstance(reduced_ops, SummedOp):
            reduced_ops = reduced_ops.collapse_summands()

        if isinstance(reduced_ops, SummedOp) and len(reduced_ops.oplist) == 1:
            return reduced_ops.oplist[0]
        else:
            return cast(OperatorBase, reduced_ops)

    def to_circuit(self) -> QuantumCircuit:
        """Returns the quantum circuit, representing the SummedOp. In the first step,
        the SummedOp is converted to MatrixOp. This is straightforward for most operators,
        but it is not supported for operators containing parametrized PrimitiveOps (in that case,
        AquaError is raised). In the next step, the MatrixOp representation of SummedOp is
        converted to circuit. In most cases, if the summands themselves are unitary operators,
        the SummedOp itself is non-unitary and can not be converted to circuit. In that case,
        ExtensionError is raised in the underlying modules.

        Returns:
            The circuit representation of the summed operator.

        Raises:
            AquaError: if SummedOp can not be converted to MatrixOp (e.g. SummedOp is composed of
            parametrized PrimitiveOps).
        """
        from qiskit.aqua.operators import MatrixOp
        matrix_op = self.to_matrix_op()
        if isinstance(matrix_op, MatrixOp):
            return matrix_op.to_circuit()
        raise AquaError("The SummedOp can not be converted to circuit, because to_matrix_op did "
                        "not return a MatrixOp.")

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """ Returns an equivalent Operator composed of only NumPy-based primitives, such as
        ``MatrixOp`` and ``VectorStateFn``. """
        accum = self.oplist[0].to_matrix_op(massive=massive)  # type: ignore
        for i in range(1, len(self.oplist)):
            accum += self.oplist[i].to_matrix_op(massive=massive)  # type: ignore

        return accum * self.coeff

    def to_legacy_op(self, massive: bool = False) -> LegacyBaseOperator:
        # We do this recursively in case there are SummedOps of PauliOps in oplist.
        legacy_ops = [op.to_legacy_op(massive=massive) for op in self.oplist]

        if not all(isinstance(op, WeightedPauliOperator) for op in legacy_ops):
            # If any Operators in oplist cannot be represented by Legacy Operators, the error
            # will be raised in the offending matrix-converted result (e.g. StateFn or ListOp)
            return self.to_matrix_op(massive=massive).to_legacy_op(massive=massive)

        if isinstance(self.coeff, ParameterExpression):
            try:
                coeff = float(self.coeff)
            except TypeError as ex:
                raise TypeError('Cannot convert Operator with unbound parameter {} to Legacy '
                                'Operator'.format(self.coeff)) from ex
        else:
            coeff = cast(float, self.coeff)

        return self.combo_fn(legacy_ops) * coeff

    def print_details(self):
        """
        Print out the operator in details.
        Returns:
            str: a formatted string describes the operator.
        """
        warnings.warn("print_details() is deprecated and will be removed in "
                      "a future release. Instead you can use .to_legacy_op() "
                      "and call print_details() on it's output",
                      DeprecationWarning)
        ret = self.to_legacy_op().print_details()
        return ret

    def equals(self, other: OperatorBase) -> bool:
        """Check if other is equal to self.

        Note:
            This is not a mathematical check for equality.
            If ``self`` and ``other`` implement the same operation but differ
            in the representation (e.g. different type of summands)
            ``equals`` will evaluate to ``False``.

        Args:
            other: The other operator to check for equality.

        Returns:
            True, if other and self are equal, otherwise False.

        Examples:
            >>> from qiskit.aqua.operators import X, Z
            >>> 2 * X == X + X
            True
            >>> X + Z == Z + X
            True
        """
        self_reduced, other_reduced = self.reduce(), other.reduce()
        if not isinstance(other_reduced, type(self_reduced)):
            return False

        # check if reduced op is still a SummedOp
        if not isinstance(self_reduced, SummedOp):
            return self_reduced == other_reduced

        self_reduced = cast(SummedOp, self_reduced)
        other_reduced = cast(SummedOp, other_reduced)
        if len(self_reduced.oplist) != len(other_reduced.oplist):
            return False

        # absorb coeffs into the operators
        if self_reduced.coeff != 1:
            self_reduced = SummedOp(
                [op * self_reduced.coeff for op in self_reduced.oplist])  # type: ignore
        if other_reduced.coeff != 1:
            other_reduced = SummedOp(
                [op * other_reduced.coeff for op in other_reduced.oplist])  # type: ignore

        # compare independent of order
        return set(self_reduced) == set(other_reduced)
