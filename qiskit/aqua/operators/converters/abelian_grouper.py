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

""" AbelianGrouper Class """

import itertools
import logging

import networkx as nx
import numpy as np

from qiskit.aqua import AquaError
from .converter_base import ConverterBase
from ..list_ops.list_op import ListOp
from ..list_ops.summed_op import SummedOp
from ..operator_base import OperatorBase
from ..primitive_ops.pauli_op import PauliOp
from ..state_fns.operator_state_fn import OperatorStateFn

logger = logging.getLogger(__name__)


class AbelianGrouper(ConverterBase):
    """
    The AbelianGrouper converts SummedOps into a sum of Abelian sums. Meaning,
    it will traverse the Operator, and when it finds a SummedOp, it will evaluate which of the
    summed sub-Operators commute with one another. It will then convert each of the groups of
    commuting Operators into their own SummedOps, and return the sum-of-commuting-SummedOps.
    This is particularly useful for cases where mutually commuting groups can be handled
    similarly, as in the case of Pauli Expectations, where commuting Paulis have the same
    diagonalizing circuit rotation, or Pauli Evolutions, where commuting Paulis can be
    diagonalized together. """

    def __init__(self, traverse: bool = True) -> None:
        """
        Args:
            traverse: Whether to convert only the Operator passed to ``convert``, or traverse
                down that Operator.
        """
        self._traverse = traverse

    def convert(self, operator: OperatorBase) -> OperatorBase:
        """ Check if operator is a SummedOp, in which case covert it into a sum of mutually
        commuting sums, or if the Operator contains sub-Operators and ``traverse`` is True,
        attempt to convert any sub-Operators.

        Args:
            operator: The Operator to attempt to convert.

        Returns:
            The converted Operator.
        """
        # pylint: disable=cyclic-import,import-outside-toplevel
        from ..evolutions.evolved_op import EvolvedOp

        if isinstance(operator, ListOp):
            if isinstance(operator, SummedOp) and all(isinstance(op, PauliOp)
                                                      for op in operator.oplist):
                # For now, we only support graphs over Paulis.
                return self.group_subops(operator)
            elif self._traverse:
                return operator.traverse(self.convert)
            else:
                return operator
        elif isinstance(operator, OperatorStateFn) and self._traverse:
            return OperatorStateFn(self.convert(operator.primitive),
                                   is_measurement=operator.is_measurement,
                                   coeff=operator.coeff)
        elif isinstance(operator, EvolvedOp) and self._traverse:
            return EvolvedOp(self.convert(operator.primitive), coeff=operator.coeff)
        else:
            return operator

    @staticmethod
    def group_subops(list_op: ListOp) -> ListOp:
        """ Given a ListOp, attempt to group into Abelian ListOps of the same type.

        Args:
            list_op: The Operator to group into Abelian groups

        Returns:
            The grouped Operator.

        Raises:
            AquaError: Any of list_op's sub-ops do not have a ``commutes`` method.
        """
        if any(not hasattr(op, 'commutes') for op in list_op.oplist):
            raise AquaError('Cannot determine Abelian groups if an Operator in list_op does not '
                            'contain a `commutes` method'.format())

        commutation_graph = nx.Graph()
        commutation_graph.add_nodes_from(list_op.oplist)
        commutation_graph.add_edges_from(filter(lambda ops: not ops[0].commutes(ops[1]),
                                                itertools.combinations(list_op.oplist, 2)))

        # Keys in coloring_dict are nodes, values are colors
        # pylint: disable=no-member
        coloring_dict = nx.coloring.greedy_color(commutation_graph, strategy='largest_first')

        groups = {}
        for operator, color in coloring_dict.items():
            groups.setdefault(color, []).append(operator)

        group_ops = [list_op.__class__(group, abelian=True) for group in groups.values()]
        if len(group_ops) == 1:
            return group_ops[0] * list_op.coeff
        return list_op.__class__(group_ops, coeff=list_op.coeff)

    @staticmethod
    def group_subops_fast(list_op: ListOp, nx_trick=False) -> ListOp:
        """ Given a ListOp, attempt to group into Abelian ListOps of the same type.
        Fast version of `group_subops` implemented using numpy.

        Args:
            list_op: The Operator to group into Abelian groups
            nx_trick: Add edges directly to nx.Graph to reduce the overhead

        Returns:
            The grouped Operator.

        Raises:
            AquaError: Any of list_op's sub-ops do not have a ``commutes`` method.
        """
        if any(not hasattr(op, 'commutes') for op in list_op.oplist):
            raise AquaError('Cannot determine Abelian groups if an Operator in list_op does not '
                            'contain a `commutes` method'.format())

        def _create_edges(list_op: ListOp):
            """
            Create edges (i,j) if i and j is not commutable.

            Returns:
                A list of pairs of indices of the operators that are not commutable
            """
            conv = {
                'I': 0,
                'X': 1,
                'Y': 2,
                'Z': 3
            }
            mat1 = np.array([[conv[e] for e in str(op.primitive)] for op in list_op], dtype=np.int8)
            mat2 = mat1[:, None]
            # i and j are commutable with TPB if c[i, j] is True
            mat3 = (((mat1 * mat2) * (mat1 - mat2)) == 0).all(axis=2)
            # return [(i, j) if c[i, j] is False and i < j]
            return zip(*np.where(np.triu(np.logical_not(mat3), k=1)))

        commutation_graph = nx.Graph()
        commutation_graph.add_nodes_from(range(len(list_op)))
        if nx_trick:
            for i, j in _create_edges(list_op):
                commutation_graph._adj[i][j] = commutation_graph._adj[j][i] = None
        else:
            commutation_graph.add_edges_from(_create_edges(list_op))

        # Keys in coloring_dict are nodes, values are colors
        # pylint: disable=no-member
        coloring_dict = nx.coloring.greedy_color(commutation_graph, strategy='largest_first')

        groups = {}
        for idx, color in coloring_dict.items():
            groups.setdefault(color, []).append(list_op[idx])

        group_ops = [list_op.__class__(group, abelian=True) for group in groups.values()]
        if len(group_ops) == 1:
            return group_ops[0] * list_op.coeff
        return list_op.__class__(group_ops, coeff=list_op.coeff)
