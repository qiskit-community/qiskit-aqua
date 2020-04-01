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

""" Test Converters """

import unittest
from test.optimization.optimization_test_case import QiskitOptimizationTestCase
from cplex import SparsePair

from qiskit.optimization import OptimizationProblem, QiskitOptimizationError
from qiskit.optimization.converters import InequalityToEqualityConverter, \
    OptimizationProblemToOperator, IntegerToBinaryConverter, PenalizeLinearEqualityConstraints


class TestConverters(QiskitOptimizationTestCase):
    """Test Converters"""

    def test_empty_problem(self):
        """ test empty problem """
        op = OptimizationProblem()
        conv = InequalityToEqualityConverter()
        op = conv.encode(op)
        conv = IntegerToBinaryConverter()
        op = conv.encode(op)
        conv = PenalizeLinearEqualityConstraints()
        op = conv.encode(op)
        conv = OptimizationProblemToOperator()
        _, shift = conv.encode(op)
        self.assertEqual(shift, 0.0)

    def test_inequality_binary(self):
        """ test inequality binary """
        op = OptimizationProblem()
        op.variables.add(names=['x', 'y', 'z'], types='B' * 3)
        op.linear_constraints.add(
            lin_expr=[SparsePair(ind=['x', 'y'], val=[1, 1]),
                      SparsePair(ind=['y', 'z'], val=[1, -1]),
                      SparsePair(ind=['z', 'x'], val=[1, 2])],
            senses=['E', 'L', 'G'],
            rhs=[1, 2, 3],
            names=['xy', 'yz', 'zx']
        )
        conv = InequalityToEqualityConverter()
        op2 = conv.encode(op)
        self.assertEqual(op.get_problem_name(), op2.get_problem_name())
        self.assertEqual(op.get_problem_type(), op2.get_problem_type())
        cst = op2.linear_constraints
        self.assertListEqual(cst.get_names(), ['xy', 'yz', 'zx'])
        self.assertListEqual(cst.get_senses(), ['E', 'E', 'E'])
        self.assertListEqual(cst.get_rhs(), [1, 2, 3])

    def test_inequality_integer(self):
        """ test inequality integer """
        op = OptimizationProblem()
        op.variables.add(names=['x', 'y', 'z'], types='I' * 3, lb=[-3] * 3, ub=[3] * 3)
        op.linear_constraints.add(
            lin_expr=[SparsePair(ind=['x', 'y'], val=[1, 1]),
                      SparsePair(ind=['y', 'z'], val=[1, -1]),
                      SparsePair(ind=['z', 'x'], val=[1, 2])],
            senses=['E', 'L', 'G'],
            rhs=[1, 2, 3],
            names=['xy', 'yz', 'zx']
        )
        conv = InequalityToEqualityConverter()
        op2 = conv.encode(op)
        self.assertEqual(op.get_problem_name(), op2.get_problem_name())
        self.assertEqual(op.get_problem_type(), op2.get_problem_type())
        cst = op2.linear_constraints
        self.assertListEqual(cst.get_names(), ['xy', 'yz', 'zx'])
        self.assertListEqual(cst.get_senses(), ['E', 'E', 'E'])
        self.assertListEqual(cst.get_rhs(), [1, 2, 3])

    def test_penalize_sense(self):
        """ test penalize sense """
        op = OptimizationProblem()
        op.variables.add(names=['x', 'y', 'z'], types='B' * 3)
        op.linear_constraints.add(
            lin_expr=[SparsePair(ind=['x', 'y'], val=[1, 1]),
                      SparsePair(ind=['y', 'z'], val=[1, -1]),
                      SparsePair(ind=['z', 'x'], val=[1, 2])],
            senses=['E', 'L', 'G'],
            rhs=[1, 2, 3],
            names=['xy', 'yz', 'zx']
        )
        self.assertEqual(op.linear_constraints.get_num(), 3)
        conv = PenalizeLinearEqualityConstraints()
        self.assertRaises(QiskitOptimizationError, lambda: conv.encode(op))

    def test_penalize_binary(self):
        """ test penalize binary """
        op = OptimizationProblem()
        op.variables.add(names=['x', 'y', 'z'], types='B' * 3)
        op.linear_constraints.add(
            lin_expr=[SparsePair(ind=['x', 'y'], val=[1, 1]),
                      SparsePair(ind=['y', 'z'], val=[1, -1])],
            senses=['E', 'E'],
            rhs=[1, 2],
            names=['xy', 'yz']
        )
        self.assertEqual(op.linear_constraints.get_num(), 2)
        conv = PenalizeLinearEqualityConstraints()
        op2 = conv.encode(op)
        self.assertEqual(op2.linear_constraints.get_num(), 0)

    def test_penalize_integer(self):
        """ test penalize integer """
        op = OptimizationProblem()
        op.variables.add(names=['x', 'y', 'z'], types='I' * 3, lb=[-3] * 3, ub=[3] * 3)
        op.linear_constraints.add(
            lin_expr=[SparsePair(ind=['x', 'y'], val=[1, 1]),
                      SparsePair(ind=['y', 'z'], val=[1, -1])],
            senses=['E', 'E'],
            rhs=[1, 2],
            names=['xy', 'yz']
        )
        self.assertEqual(op.linear_constraints.get_num(), 2)
        conv = PenalizeLinearEqualityConstraints()
        op2 = conv.encode(op)
        self.assertEqual(op2.linear_constraints.get_num(), 0)

    def test_integer_to_binary(self):
        """ test integer to binary """
        op = OptimizationProblem()
        op.variables.add(names=['x', 'y', 'z'], types='BIC', lb=[0, 0, 0], ub=[1, 5, 10])
        self.assertEqual(op.variables.get_num(), 3)
        conv = IntegerToBinaryConverter()
        op2 = conv.encode(op)
        print(op2.variables.get_num())
        names = op2.variables.get_names()
        self.assertIn('x', names)
        self.assertIn('z', names)
        variables = op2.variables
        self.assertEqual(variables.get_lower_bounds('x'), 0.0)
        self.assertEqual(variables.get_lower_bounds('z'), 0.0)
        self.assertEqual(variables.get_upper_bounds('x'), 1.0)
        self.assertEqual(variables.get_upper_bounds('z'), 10.0)
        self.assertListEqual(variables.get_types(['x', 'z']), ['B', 'C'])


if __name__ == '__main__':
    unittest.main()
