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

""" Expectation Algorithm Base """

import logging
import itertools
import networkx as nx

from ..operator_base import OperatorBase
from ..combo_operators import ListOp, SummedOp
from ..state_functions import OperatorStateFn
from ..primitive_operators import PauliOp
from .converter_base import ConverterBase

logger = logging.getLogger(__name__)


class AbelianGrouper(ConverterBase):
    """ Expectation Algorithm Base """
    def __init__(self, traverse=True):
        self._traverse = traverse

    def convert(self, operator: OperatorBase) -> OperatorBase:
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .. import EvolvedOp

        if isinstance(operator, ListOp):
            if isinstance(operator, SummedOp) and all([isinstance(op, PauliOp)
                                                       for op in operator.oplist]):
                # For now, we only support graphs over Paulis.
                return self.group_paulis(operator)
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

    def group_paulis(self, op_vec: ListOp) -> ListOp:
        """ group paulis """
        commutation_graph = nx.Graph()
        commutation_graph.add_nodes_from(op_vec.oplist)
        commutation_graph.add_edges_from(filter(lambda ops: not ops[0].commutes(ops[1]),
                                                itertools.combinations(op_vec.oplist, 2)))

        # Keys in coloring_dict are nodes, values are colors
        # pylint: disable=no-member
        coloring_dict = nx.coloring.greedy_color(commutation_graph, strategy='largest_first')

        groups = {}
        for op, color in coloring_dict.items():
            groups.setdefault(color, []).append(op)

        group_ops = [op_vec.__class__(group, abelian=True) for group in groups.values()]
        if len(group_ops) == 1:
            return group_ops[0] * op_vec.coeff
        else:
            return op_vec.__class__(group_ops, coeff=op_vec.coeff)
