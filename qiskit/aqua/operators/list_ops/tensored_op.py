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

""" TensoredOp Class """

from typing import List, Union, cast
from functools import reduce, partial
import numpy as np
from sympy.combinatorics import Permutation

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.quantum_info import Pauli

from ..operator_base import OperatorBase
from .list_op import ListOp
from ... import AquaError


class TensoredOp(ListOp):
    """  A class for lazily representing tensor products of Operators. Often Operators cannot be
    efficiently tensored to one another, but may be manipulated further so that they can be
    later. This class holds logic to indicate that the Operators in ``oplist`` are meant to
    be tensored together, and therefore if they reach a point in which they can be, such as after
    conversion to QuantumCircuits, they can be reduced by tensor product. """
    def __init__(self,
                 oplist: List[OperatorBase],
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 abelian: bool = False) -> None:
        """
        Args:
            oplist: The Operators being tensored.
            coeff: A coefficient multiplying the operator
            abelian: Indicates whether the Operators in ``oplist`` are known to mutually commute.
        """
        super().__init__(oplist,
                         combo_fn=partial(reduce, np.kron),
                         coeff=coeff,
                         abelian=abelian)

    @property
    def num_qubits(self) -> int:
        return sum([op.num_qubits for op in self.oplist])

    @property
    def distributive(self) -> bool:
        return False

    def expand_to_dim(self, num_qubits: int) -> 'TensoredOp':
        """ Appends PauliOp, with Pauli I as primitive, to ``oplist``.
        Choice of Pauli as identity is arbitrary and could be substituted for other
        PrimitiveOp.identity.
        Returns:
            identity operator represented by Pauli(label='I'*num_qubit)
        """
        from qiskit.aqua.operators import PauliOp
        return TensoredOp(self.oplist + [PauliOp(Pauli(label='I' * num_qubits))])

    def permute(self, permutation: List[int]) -> 'ListOp':
        r"""
        Permute the qubits of the operator.

        Args:
            permutation: A list defining where each qubit should be permuted. The qubit at index
                j should be permuted to position permutation[j].

        Returns:
            A new ListOp containing the permuted circuit.
        Raises:
            AquaError: if indices does not define a new index for each qubit.
        """
        new_self = self
        circuit_size = max(permutation) + 1

        if self.num_qubits != len(permutation):
            raise AquaError("New index must be defined for each qubit of the operator.")
        if self.num_qubits < circuit_size:
            # pad the operator with identities
            new_self = self.expand_to_dim(circuit_size - self.num_qubits)
        qc = QuantumCircuit(circuit_size)
        # extend the indices to match the size of the circuit
        indices = list(filter(lambda x: x not in indices, range(circuit_size))) + permutation

        # decompose permutation into sequence of transpositions
        transpositions = Permutation(indices).transpositions()
        for trans in transpositions:
            qc.swap(trans[0], trans[1])

        from qiskit.aqua.operators import CircuitOp, MatrixOp

        matrix = CircuitOp(qc).to_matrix()
        return MatrixOp(matrix.transpose()) @ new_self @ MatrixOp(matrix)  # type: ignore

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if isinstance(other, TensoredOp):
            return TensoredOp(self.oplist + other.oplist, coeff=self.coeff * other.coeff)
        return TensoredOp(self.oplist + [other], coeff=self.coeff)

    # TODO eval should partial trace the input into smaller StateFns each of size
    #  op.num_qubits for each op in oplist. Right now just works through matmul.
    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        return cast(Union[OperatorBase, float, complex], self.to_matrix_op().eval(front=front))

    # Try collapsing list or trees of tensor products.
    # TODO do this smarter
    def reduce(self) -> OperatorBase:
        reduced_ops = [op.reduce() for op in self.oplist]
        reduced_ops = reduce(lambda x, y: x.tensor(y), reduced_ops) * self.coeff
        if isinstance(reduced_ops, ListOp) and len(reduced_ops.oplist) == 1:
            return reduced_ops.oplist[0]
        else:
            return cast(OperatorBase, reduced_ops)
