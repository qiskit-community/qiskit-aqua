# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

CPXPROB_LP = 0
CPXPROB_MILP = 1
CPXPROB_FIXEDMILP = 3
CPXPROB_NODELP = 4
CPXPROB_QP = 5
CPXPROB_MIQP = 7
CPXPROB_FIXEDMIQP = 8
CPXPROB_NODEQP = 9
CPXPROB_QCP = 10
CPXPROB_MIQCP = 11
CPXPROB_NODEQCP = 12


class ProblemType(object):
    """
    Types of problems the OptimizationProblem class can encapsulate.
    These types are compatible with those of IBM ILOG CPLEX.
    For explanations of the problem types, see those topics in the
    CPLEX User's Manual in the topic titled Continuous Optimization
    for LP, QP, and QCP or the topic titled Discrete Optimization
    for MILP, FIXEDMILP, NODELP, NODEQP, MIQCP, NODEQCP.
    """
    LP = CPXPROB_LP
    MILP = CPXPROB_MILP
    fixed_MILP = CPXPROB_FIXEDMILP
    node_LP = CPXPROB_NODELP
    QP = CPXPROB_QP
    MIQP = CPXPROB_MIQP
    fixed_MIQP = CPXPROB_FIXEDMIQP
    node_QP = CPXPROB_NODEQP
    QCP = CPXPROB_QCP
    MIQCP = CPXPROB_MIQCP
    node_QCP = CPXPROB_NODEQCP

    def __getitem__(self, item):
        """Converts a constant to a string.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> op.problem_type.LP
        0
        >>> op.problem_type[0]
        'LP'
        """
        if item == CPXPROB_LP:
            return 'LP'
        if item == CPXPROB_MILP:
            return 'MILP'
        if item == CPXPROB_FIXEDMILP:
            return 'fixed_MILP'
        if item == CPXPROB_NODELP:
            return 'node_LP'
        if item == CPXPROB_QP:
            return 'QP'
        if item == CPXPROB_MIQP:
            return 'MIQP'
        if item == CPXPROB_FIXEDMIQP:
            return 'fixed_MIQP'
        if item == CPXPROB_NODEQP:
            return 'node_QP'
        if item == CPXPROB_QCP:
            return 'QCP'
        if item == CPXPROB_MIQCP:
            return 'MIQCP'
        if item == CPXPROB_NODEQCP:
            return 'node_QCP'
