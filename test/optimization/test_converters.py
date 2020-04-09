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
import logging

from qiskit.optimization import QuadraticProgram, QiskitOptimizationError
from qiskit.optimization.results import OptimizationResult
from qiskit.optimization.converters import InequalityToEqualityConverter, \
    QuadraticProgramToOperator, IntegerToBinaryConverter, PenalizeLinearEqualityConstraints
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli

logger = logging.getLogger(__name__)

_HAS_CPLEX = False
try:
    from cplex import SparsePair
    _HAS_CPLEX = True
except ImportError:
    logger.info('CPLEX is not installed.')

QUBIT_OP_MAXIMIZE_SAMPLE = WeightedPauliOperator(
    paulis=[[(-199999.5+0j), Pauli(z=[True, False, False, False],
                                   x=[False, False, False, False])],
            [(-399999.5+0j), Pauli(z=[False, True, False, False],
                                   x=[False, False, False, False])],
            [(-599999.5+0j), Pauli(z=[False, False, True, False],
                                   x=[False, False, False, False])],
            [(-799999.5+0j), Pauli(z=[False, False, False, True],
                                   x=[False, False, False, False])],
            [(100000+0j), Pauli(z=[True, True, False, False],
                                x=[False, False, False, False])],
            [(150000+0j), Pauli(z=[True, False, True, False],
                                x=[False, False, False, False])],
            [(200000+0j), Pauli(z=[True, False, False, True],
                                x=[False, False, False, False])],
            [(300000+0j), Pauli(z=[False, True, True, False],
                                x=[False, False, False, False])],
            [(400000+0j), Pauli(z=[False, True, False, True],
                                x=[False, False, False, False])],
            [(600000+0j), Pauli(z=[False, False, True, True],
                                x=[False, False, False, False])]])
OFFSET_MAXIMIZE_SAMPLE = 1149998


class TestConverters(QiskitOptimizationTestCase):
    """Test Converters"""

    def setUp(self) -> None:
        super().setUp()
        if not _HAS_CPLEX:
            self.skipTest('CPLEX is not installed.')

    def test_empty_problem(self):
        """ Test empty problem """
        op = QuadraticProgram()
        conv = InequalityToEqualityConverter()
        op = conv.encode(op)
        conv = IntegerToBinaryConverter()
        op = conv.encode(op)
        conv = PenalizeLinearEqualityConstraints()
        op = conv.encode(op)
        conv = QuadraticProgramToOperator()
        _, shift = conv.encode(op)
        self.assertEqual(shift, 0.0)

    def test_valid_variable_type(self):
        """Validate the types of the variables for QuadraticProgramToOperator."""
        # Integer variable
        with self.assertRaises(QiskitOptimizationError):
            op = QuadraticProgram()
            op.variables.add(names=['x'], types='I')
            conv = QuadraticProgramToOperator()
            _ = conv.encode(op)
        # Continuous variable
        with self.assertRaises(QiskitOptimizationError):
            op = QuadraticProgram()
            op.variables.add(names=['x'], types='C')
            conv = QuadraticProgramToOperator()
            _ = conv.encode(op)
        # Semi-Continuous variable
        with self.assertRaises(QiskitOptimizationError):
            op = QuadraticProgram()
            op.variables.add(names=['x'], types='S')
            conv = QuadraticProgramToOperator()
            _ = conv.encode(op)
        # Semi-Integer variable
        with self.assertRaises(QiskitOptimizationError):
            op = QuadraticProgram()
            op.variables.add(names=['x'], types='N')
            conv = QuadraticProgramToOperator()
            _ = conv.encode(op)
        # validate the types of the variables for InequalityToEqualityConverter
        # Semi-Continuous variable
        with self.assertRaises(QiskitOptimizationError):
            op = QuadraticProgram()
            op.variables.add(names=['x'], types='S')
            conv = InequalityToEqualityConverter()
            _ = conv.encode(op)
        # Semi-Integer variable
        with self.assertRaises(QiskitOptimizationError):
            op = QuadraticProgram()
            op.variables.add(names=['x'], types='N')
            conv = InequalityToEqualityConverter()
            _ = conv.encode(op)

    def test_inequality_binary(self):
        """ Test InequalityToEqualityConverter with binary variables """
        op = QuadraticProgram()
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
        var = op2.variables
        self.assertListEqual(var.get_lower_bounds(3, 4), [0, 0])
        self.assertListEqual(var.get_upper_bounds(3, 4), [3, 0])

    def test_inequality_integer(self):
        """ Test InequalityToEqualityConverter with integer variables """
        op = QuadraticProgram()
        op.variables.add(names=['x', 'y', 'z'],
                         types='I' * 3, lb=[-3] * 3, ub=[3] * 3)
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
        var = op2.variables
        self.assertListEqual(var.get_lower_bounds(3, 4), [0, 0])
        self.assertListEqual(var.get_upper_bounds(3, 4), [8, 6])

    def test_inequality_mode_integer(self):
        """ Test integer mode of InequalityToEqualityConverter() """
        op = QuadraticProgram()
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
        op2 = conv.encode(op, mode='integer')
        var = op2.variables
        self.assertListEqual(var.get_types(3, 4), ['I', 'I'])

    def test_inequality_mode_continuous(self):
        """ Test continuous mode of InequalityToEqualityConverter() """
        op = QuadraticProgram()
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
        op2 = conv.encode(op, mode='continuous')
        var = op2.variables
        self.assertListEqual(var.get_types(3, 4), ['C', 'C'])

    def test_inequality_mode_auto(self):
        """ Test auto mode of InequalityToEqualityConverter() """
        op = QuadraticProgram()
        op.variables.add(names=['x', 'y', 'z'], types='B' * 3)
        op.linear_constraints.add(
            lin_expr=[SparsePair(ind=['x', 'y'], val=[1, 1]),
                      SparsePair(ind=['y', 'z'], val=[1, -1]),
                      SparsePair(ind=['z', 'x'], val=[1.1, 2.2])],
            senses=['E', 'L', 'G'],
            rhs=[1, 2, 3.3],
            names=['xy', 'yz', 'zx']
        )
        conv = InequalityToEqualityConverter()
        op2 = conv.encode(op, mode='auto')
        var = op2.variables
        self.assertListEqual(var.get_types(3, 4), ['I', 'C'])

    def test_penalize_sense(self):
        """ Test PenalizeLinearEqualityConstraints with senses """
        op = QuadraticProgram()
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
        with self.assertRaises(QiskitOptimizationError):
            conv.encode(op)

    def test_penalize_binary(self):
        """ Test PenalizeLinearEqualityConstraints with binary variables """
        op = QuadraticProgram()
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
        """ Test PenalizeLinearEqualityConstraints with integer variables """
        op = QuadraticProgram()
        op.variables.add(names=['x', 'y', 'z'],
                         types='I' * 3, lb=[-3] * 3, ub=[3] * 3)
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
        """ Test integer to binary """
        op = QuadraticProgram()
        op.variables.add(names=['x', 'y', 'z'], types='BIC',
                         lb=[0, 0, 0], ub=[1, 6, 10])
        op.objective.set_linear([('x', 1), ('y', 2), ('z', 1)])
        op.linear_constraints.add(
            lin_expr=[SparsePair(ind=['x', 'y', 'z'], val=[1, 3, 1])],
            senses=['L'],
            rhs=[10],
            names=['xyz']
        )
        self.assertEqual(op.variables.get_num(), 3)
        conv = IntegerToBinaryConverter()
        op2 = conv.encode(op)
        names = op2.variables.get_names()
        self.assertIn('x', names)
        self.assertIn('z', names)
        variables = op2.variables
        self.assertEqual(variables.get_lower_bounds('x'), 0.0)
        self.assertEqual(variables.get_lower_bounds('z'), 0.0)
        self.assertEqual(variables.get_upper_bounds('x'), 1.0)
        self.assertEqual(variables.get_upper_bounds('z'), 10.0)
        self.assertListEqual(variables.get_types(['x', 'y@0', 'y@1', 'y@2', 'z']),
                             ['B', 'B', 'B', 'B', 'C'])
        self.assertListEqual(op2.objective.get_linear(['y@0', 'y@1', 'y@2']), [2, 4, 6])
        self.assertListEqual(op2.linear_constraints.get_rows()[0].val, [1, 3, 6, 9, 1])

    def test_binary_to_integer(self):
        """ Test binary to integer """
        op = QuadraticProgram()
        op.variables.add(names=['x', 'y', 'z'], types='BIB', lb=[
            0, 0, 0], ub=[1, 7, 1])
        op.objective.set_linear([('x', 2), ('y', 1), ('z', 1)])
        op.linear_constraints.add(
            lin_expr=[SparsePair(ind=['x', 'y', 'z'], val=[1, 1, 1])],
            senses=['L'],
            rhs=[7],
            names=['xyz']
        )
        op.objective.set_sense(-1)
        conv = IntegerToBinaryConverter()
        _ = conv.encode(op)
        result = OptimizationResult(x=[1, 0., 1, 1, 0], fval=8)
        new_result = conv.decode(result)
        self.assertListEqual(new_result.x, [1, 6, 0])
        self.assertEqual(new_result.fval, 8)

    def test_optimizationproblem_to_operator(self):
        """ Test optimization problem to operators"""
        op = QuadraticProgram()
        op.variables.add(names=['a', 'b', 'c', 'd'], types='B'*4)
        op.objective.set_linear([('a', 1), ('b', 1), ('c', 1), ('d', 1)])
        op.linear_constraints.add(
            lin_expr=[SparsePair(ind=['a', 'b', 'c', 'd'], val=[1, 2, 3, 4])],
            senses=['E'],
            rhs=[3],
            names=['abcd']
        )
        op.objective.set_sense(-1)
        penalize = PenalizeLinearEqualityConstraints()
        op2ope = QuadraticProgramToOperator()
        op2 = penalize.encode(op)
        qubitop, offset = op2ope.encode(op2)
        self.assertListEqual(qubitop.paulis, QUBIT_OP_MAXIMIZE_SAMPLE.paulis)
        self.assertEqual(offset, OFFSET_MAXIMIZE_SAMPLE)

    def test_quadratic_constraints(self):
        """ Test quadratic constraints"""
        # IntegerToBinaryConverter
        with self.assertRaises(QiskitOptimizationError):
            op = QuadraticProgram()
            op.variables.add(names=['x', 'y'])
            l_expr = SparsePair(ind=['x'], val=[1.0])
            q_expr = [['x'], ['y'], [1]]
            op.quadratic_constraints.add(name=str(1), lin_expr=l_expr, quad_expr=q_expr)
            conv = IntegerToBinaryConverter()
            _ = conv.encode(op)
        # InequalityToEqualityConverter
        with self.assertRaises(QiskitOptimizationError):
            op = QuadraticProgram()
            op.variables.add(names=['x', 'y'])
            l_expr = SparsePair(ind=['x'], val=[1.0])
            q_expr = [['x'], ['y'], [1]]
            op.quadratic_constraints.add(name=str(1), lin_expr=l_expr, quad_expr=q_expr)
            conv = InequalityToEqualityConverter()
            _ = conv.encode(op)


if __name__ == '__main__':
    unittest.main()
