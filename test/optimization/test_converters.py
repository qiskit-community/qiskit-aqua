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

import logging
import unittest
from test.optimization.optimization_test_case import QiskitOptimizationTestCase

import numpy as np
from docplex.mp.model import Model
from qiskit.aqua import MissingOptionalLibraryError
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.aqua.operators import Z, I
from qiskit.optimization import QuadraticProgram, QiskitOptimizationError
from qiskit.optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer, ADMMOptimizer
from qiskit.optimization.algorithms.admm_optimizer import ADMMParameters
from qiskit.optimization.converters import (InequalityToEquality, IntegerToBinary,
                                            LinearEqualityToPenalty)
from qiskit.optimization.problems import Constraint, Variable

logger = logging.getLogger(__name__)

QUBIT_OP_MAXIMIZE_SAMPLE = (
        -199999.5 * (I ^ I ^ I ^ Z)
        + -399999.5 * (I ^ I ^ Z ^ I)
        + -599999.5 * (I ^ Z ^ I ^ I)
        + -799999.5 * (Z ^ I ^ I ^ I)
        + 100000 * (I ^ I ^ Z ^ Z)
        + 150000 * (I ^ Z ^ I ^ Z)
        + 300000 * (I ^ Z ^ Z ^ I)
        + 200000 * (Z ^ I ^ I ^ Z)
        + 400000 * (Z ^ I ^ Z ^ I)
        + 600000 * (Z ^ Z ^ I ^ I)
)
OFFSET_MAXIMIZE_SAMPLE = 1149998


class TestConverters(QiskitOptimizationTestCase):
    """Test Converters"""

    def test_empty_problem(self):
        """ Test empty problem """
        op = QuadraticProgram()
        conv = InequalityToEquality()
        op = conv.convert(op)
        conv = IntegerToBinary()
        op = conv.convert(op)
        conv = LinearEqualityToPenalty()
        op = conv.convert(op)
        _, shift = op.to_ising()
        self.assertEqual(shift, 0.0)

    def test_valid_variable_type(self):
        """Validate the types of the variables for QuadraticProgram.to_ising."""
        # Integer variable
        with self.assertRaises(QiskitOptimizationError):
            op = QuadraticProgram()
            op.integer_var(0, 10, "int_var")
            _ = op.to_ising()
        # Continuous variable
        with self.assertRaises(QiskitOptimizationError):
            op = QuadraticProgram()
            op.continuous_var(0, 10, "continuous_var")
            _ = op.to_ising()

    def test_inequality_binary(self):
        """ Test InequalityToEqualityConverter with binary variables """
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name='x{}'.format(i))
        # Linear constraints
        linear_constraint = {'x0': 1, 'x1': 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {'x1': 1, 'x2': -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, 'x1x2')
        linear_constraint = {'x0': 1, 'x2': 3}
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 2, 'x0x2')
        # Quadratic constraints
        quadratic = {('x0', 'x1'): 1, ('x1', 'x2'): 2}
        op.quadratic_constraint({}, quadratic, Constraint.Sense.LE, 3, 'x0x1_x1x2LE')
        quadratic = {('x0', 'x1'): 3, ('x1', 'x2'): 4}
        op.quadratic_constraint({}, quadratic, Constraint.Sense.GE, 3, 'x0x1_x1x2GE')
        # Convert inequality constraints into equality constraints
        conv = InequalityToEquality()
        op2 = conv.convert(op)
        self.assertListEqual([v.name for v in op2.variables],
                             ['x0', 'x1', 'x2', 'x1x2@int_slack', 'x0x2@int_slack',
                              'x0x1_x1x2LE@int_slack', 'x0x1_x1x2GE@int_slack'])
        # Check names and objective senses
        self.assertEqual(op.name, op2.name)
        self.assertEqual(op.objective.sense, op2.objective.sense)
        # For linear constraints
        lst = [
            op2.linear_constraints[0].linear.to_dict()[0],
            op2.linear_constraints[0].linear.to_dict()[1],
        ]
        self.assertListEqual(lst, [1, 1])
        self.assertEqual(op2.linear_constraints[0].sense, Constraint.Sense.EQ)
        lst = [
            op2.linear_constraints[1].linear.to_dict()[1],
            op2.linear_constraints[1].linear.to_dict()[2],
            op2.linear_constraints[1].linear.to_dict()[3],
        ]
        self.assertListEqual(lst, [1, -1, 1])
        lst = [op2.variables[3].lowerbound, op2.variables[3].upperbound]
        self.assertListEqual(lst, [0, 3])
        self.assertEqual(op2.linear_constraints[1].sense, Constraint.Sense.EQ)
        lst = [
            op2.linear_constraints[2].linear.to_dict()[0],
            op2.linear_constraints[2].linear.to_dict()[2],
            op2.linear_constraints[2].linear.to_dict()[4],
        ]
        self.assertListEqual(lst, [1, 3, -1])
        lst = [op2.variables[4].lowerbound, op2.variables[4].upperbound]
        self.assertListEqual(lst, [0, 2])
        self.assertEqual(op2.linear_constraints[2].sense, Constraint.Sense.EQ)
        # For quadratic constraints
        lst = [
            op2.quadratic_constraints[0].quadratic.to_dict()[(0, 1)],
            op2.quadratic_constraints[0].quadratic.to_dict()[(1, 2)],
            op2.quadratic_constraints[0].linear.to_dict()[5],
        ]
        self.assertListEqual(lst, [1, 2, 1])
        lst = [op2.variables[5].lowerbound, op2.variables[5].upperbound]
        self.assertListEqual(lst, [0, 3])
        lst = [
            op2.quadratic_constraints[1].quadratic.to_dict()[(0, 1)],
            op2.quadratic_constraints[1].quadratic.to_dict()[(1, 2)],
            op2.quadratic_constraints[1].linear.to_dict()[6],
        ]
        self.assertListEqual(lst, [3, 4, -1])
        lst = [op2.variables[6].lowerbound, op2.variables[6].upperbound]
        self.assertListEqual(lst, [0, 4])

        new_x = conv.interpret(np.arange(7))
        np.testing.assert_array_almost_equal(new_x, np.arange(3))

    def test_inequality_integer(self):
        """ Test InequalityToEqualityConverter with integer variables """
        op = QuadraticProgram()
        for i in range(3):
            op.integer_var(name='x{}'.format(i), lowerbound=-3, upperbound=3)
        # Linear constraints
        linear_constraint = {'x0': 1, 'x1': 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {'x1': 1, 'x2': -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, 'x1x2')
        linear_constraint = {'x0': 1, 'x2': 3}
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 2, 'x0x2')
        # Quadratic constraints
        quadratic = {('x0', 'x1'): 1, ('x1', 'x2'): 2}
        op.quadratic_constraint({}, quadratic, Constraint.Sense.LE, 3, 'x0x1_x1x2LE')
        quadratic = {('x0', 'x1'): 3, ('x1', 'x2'): 4}
        op.quadratic_constraint({}, quadratic, Constraint.Sense.GE, 3, 'x0x1_x1x2GE')
        conv = InequalityToEquality()
        op2 = conv.convert(op)
        self.assertListEqual([v.name for v in op2.variables],
                             ['x0', 'x1', 'x2', 'x1x2@int_slack', 'x0x2@int_slack',
                              'x0x1_x1x2LE@int_slack', 'x0x1_x1x2GE@int_slack'])
        # For linear constraints
        lst = [
            op2.linear_constraints[0].linear.to_dict()[0],
            op2.linear_constraints[0].linear.to_dict()[1],
        ]
        self.assertListEqual(lst, [1, 1])
        self.assertEqual(op2.linear_constraints[0].sense, Constraint.Sense.EQ)
        lst = [
            op2.linear_constraints[1].linear.to_dict()[1],
            op2.linear_constraints[1].linear.to_dict()[2],
            op2.linear_constraints[1].linear.to_dict()[3],
        ]
        self.assertListEqual(lst, [1, -1, 1])
        lst = [op2.variables[3].lowerbound, op2.variables[3].upperbound]
        self.assertListEqual(lst, [0, 8])
        self.assertEqual(op2.linear_constraints[1].sense, Constraint.Sense.EQ)
        lst = [
            op2.linear_constraints[2].linear.to_dict()[0],
            op2.linear_constraints[2].linear.to_dict()[2],
            op2.linear_constraints[2].linear.to_dict()[4],
        ]
        self.assertListEqual(lst, [1, 3, -1])
        lst = [op2.variables[4].lowerbound, op2.variables[4].upperbound]
        self.assertListEqual(lst, [0, 10])
        self.assertEqual(op2.linear_constraints[2].sense, Constraint.Sense.EQ)
        # For quadratic constraints
        lst = [
            op2.quadratic_constraints[0].quadratic.to_dict()[(0, 1)],
            op2.quadratic_constraints[0].quadratic.to_dict()[(1, 2)],
            op2.quadratic_constraints[0].linear.to_dict()[5],
        ]
        self.assertListEqual(lst, [1, 2, 1])
        lst = [op2.variables[5].lowerbound, op2.variables[5].upperbound]
        self.assertListEqual(lst, [0, 30])
        lst = [
            op2.quadratic_constraints[1].quadratic.to_dict()[(0, 1)],
            op2.quadratic_constraints[1].quadratic.to_dict()[(1, 2)],
            op2.quadratic_constraints[1].linear.to_dict()[6],
        ]
        self.assertListEqual(lst, [3, 4, -1])
        lst = [op2.variables[6].lowerbound, op2.variables[6].upperbound]
        self.assertListEqual(lst, [0, 60])

        new_x = conv.interpret(np.arange(7))
        np.testing.assert_array_almost_equal(new_x, np.arange(3))

    def test_inequality_mode_integer(self):
        """ Test integer mode of InequalityToEqualityConverter() """
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name='x{}'.format(i))
        # Linear constraints
        linear_constraint = {'x0': 1, 'x1': 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {'x1': 1, 'x2': -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, 'x1x2')
        linear_constraint = {'x0': 1, 'x2': 3}
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 2, 'x0x2')
        conv = InequalityToEquality(mode='integer')
        op2 = conv.convert(op)
        lst = [op2.variables[3].vartype, op2.variables[4].vartype]
        self.assertListEqual(lst, [Variable.Type.INTEGER, Variable.Type.INTEGER])

    def test_inequality_mode_continuous(self):
        """ Test continuous mode of InequalityToEqualityConverter() """
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name='x{}'.format(i))
        # Linear constraints
        linear_constraint = {'x0': 1, 'x1': 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {'x1': 1, 'x2': -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, 'x1x2')
        linear_constraint = {'x0': 1, 'x2': 3}
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 2, 'x0x2')
        conv = InequalityToEquality(mode='continuous')
        op2 = conv.convert(op)
        lst = [op2.variables[3].vartype, op2.variables[4].vartype]
        self.assertListEqual(lst, [Variable.Type.CONTINUOUS, Variable.Type.CONTINUOUS])

    def test_inequality_mode_auto(self):
        """ Test auto mode of InequalityToEqualityConverter() """
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name='x{}'.format(i))
        # Linear constraints
        linear_constraint = {'x0': 1, 'x1': 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {'x1': 1, 'x2': -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, 'x1x2')
        linear_constraint = {'x0': 1.1, 'x2': 2.2}
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 3.3, 'x0x2')
        conv = InequalityToEquality(mode='auto')
        op2 = conv.convert(op)
        lst = [op2.variables[3].vartype, op2.variables[4].vartype]
        self.assertListEqual(lst, [Variable.Type.INTEGER, Variable.Type.CONTINUOUS])

    def test_penalize_sense(self):
        """ Test PenalizeLinearEqualityConstraints with senses """
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name='x{}'.format(i))
        # Linear constraints
        linear_constraint = {'x0': 1, 'x1': 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {'x1': 1, 'x2': -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, 'x1x2')
        linear_constraint = {'x0': 1, 'x2': 3}
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 2, 'x0x2')
        self.assertEqual(op.get_num_linear_constraints(), 3)
        conv = LinearEqualityToPenalty()
        with self.assertRaises(QiskitOptimizationError):
            conv.convert(op)

    def test_penalize_binary(self):
        """ Test PenalizeLinearEqualityConstraints with binary variables """
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name='x{}'.format(i))
        # Linear constraints
        linear_constraint = {'x0': 1, 'x1': 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {'x1': 1, 'x2': -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 2, 'x1x2')
        linear_constraint = {'x0': 1, 'x2': 3}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 2, 'x0x2')
        self.assertEqual(op.get_num_linear_constraints(), 3)
        conv = LinearEqualityToPenalty()
        op2 = conv.convert(op)
        self.assertEqual(op2.get_num_linear_constraints(), 0)

        new_x = conv.interpret(np.arange(3))
        np.testing.assert_array_almost_equal(new_x, np.arange(3))

    def test_penalize_integer(self):
        """ Test PenalizeLinearEqualityConstraints with integer variables """
        op = QuadraticProgram()
        for i in range(3):
            op.integer_var(name='x{}'.format(i), lowerbound=-3, upperbound=3)
        # Linear constraints
        linear_constraint = {'x0': 1, 'x1': 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {'x1': 1, 'x2': -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 2, 'x1x2')
        linear_constraint = {'x0': 1, 'x2': -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x2')
        op.minimize(constant=3, linear={'x0': 1}, quadratic={('x1', 'x2'): 2})
        self.assertEqual(op.get_num_linear_constraints(), 3)
        conv = LinearEqualityToPenalty()
        op2 = conv.convert(op)
        self.assertEqual(op2.get_num_linear_constraints(), 0)

        new_x = conv.interpret([0, 1, -1])
        np.testing.assert_array_almost_equal(new_x, [0, 1, -1])

    def test_integer_to_binary(self):
        """ Test integer to binary """
        op = QuadraticProgram()
        for i in range(0, 2):
            op.binary_var(name='x{}'.format(i))
        op.integer_var(name='x2', lowerbound=0, upperbound=5)
        linear = {}
        for i, x in enumerate(op.variables):
            linear[x.name] = i + 1
        op.maximize(0, linear, {})
        conv = IntegerToBinary()
        op2 = conv.convert(op)
        self.assertEqual(op2.get_num_vars(), 5)
        self.assertListEqual([x.vartype for x in op2.variables], [Variable.Type.BINARY] * 5)
        self.assertListEqual([x.name for x in op2.variables], ['x0', 'x1', 'x2@0', 'x2@1', 'x2@2'])
        dct = op2.objective.linear.to_dict()
        self.assertEqual(dct[2], 3)
        self.assertEqual(dct[3], 6)
        self.assertEqual(dct[4], 6)

    def test_binary_to_integer(self):
        """ Test binary to integer """
        op = QuadraticProgram()
        for i in range(0, 2):
            op.binary_var(name='x{}'.format(i))
        op.integer_var(name='x2', lowerbound=0, upperbound=5)
        linear = {'x0': 1, 'x1': 2, 'x2': 1}
        op.maximize(0, linear, {})
        linear = {}
        for x in op.variables:
            linear[x.name] = 1
        op.linear_constraint(linear, Constraint.Sense.EQ, 6, 'x0x1x2')
        conv = IntegerToBinary()
        _ = conv.convert(op)
        new_x = conv.interpret([0, 1, 1, 1, 1])
        np.testing.assert_array_almost_equal(new_x, [0, 1, 5])

    def test_optimizationproblem_to_ising(self):
        """ Test optimization problem to operators"""
        op = QuadraticProgram()
        for i in range(4):
            op.binary_var(name='x{}'.format(i))
        linear = {}
        for x in op.variables:
            linear[x.name] = 1
        op.maximize(0, linear, {})
        linear = {}
        for i, x in enumerate(op.variables):
            linear[x.name] = i + 1
        op.linear_constraint(linear, Constraint.Sense.EQ, 3, 'sum1')
        penalize = LinearEqualityToPenalty(penalty=1e5)
        op2 = penalize.convert(op)
        qubitop, offset = op2.to_ising()
        self.assertEqual(qubitop, QUBIT_OP_MAXIMIZE_SAMPLE)
        self.assertEqual(offset, OFFSET_MAXIMIZE_SAMPLE)

    def test_ising_to_quadraticprogram_linear(self):
        """ Test optimization problem to operators with linear=True"""
        op = QUBIT_OP_MAXIMIZE_SAMPLE
        offset = OFFSET_MAXIMIZE_SAMPLE

        quadratic = QuadraticProgram()
        quadratic.from_ising(op, offset, linear=True)

        self.assertEqual(quadratic.get_num_vars(), 4)
        self.assertEqual(quadratic.get_num_linear_constraints(), 0)
        self.assertEqual(quadratic.get_num_quadratic_constraints(), 0)
        self.assertEqual(quadratic.objective.sense, quadratic.objective.Sense.MINIMIZE)
        self.assertAlmostEqual(quadratic.objective.constant, 900000)

        linear_matrix = np.zeros((1, 4))
        linear_matrix[0, 0] = -500001
        linear_matrix[0, 1] = -800001
        linear_matrix[0, 2] = -900001
        linear_matrix[0, 3] = -800001

        quadratic_matrix = np.zeros((4, 4))
        quadratic_matrix[0, 1] = 400000
        quadratic_matrix[0, 2] = 600000
        quadratic_matrix[1, 2] = 1200000
        quadratic_matrix[0, 3] = 800000
        quadratic_matrix[1, 3] = 1600000
        quadratic_matrix[2, 3] = 2400000

        np.testing.assert_array_almost_equal(
            quadratic.objective.linear.coefficients.toarray(), linear_matrix
        )
        np.testing.assert_array_almost_equal(
            quadratic.objective.quadratic.coefficients.toarray(), quadratic_matrix
        )

    def test_ising_to_quadraticprogram_quadratic(self):
        """ Test optimization problem to operators with linear=False"""
        op = QUBIT_OP_MAXIMIZE_SAMPLE
        offset = OFFSET_MAXIMIZE_SAMPLE

        quadratic = QuadraticProgram()
        quadratic.from_ising(op, offset, linear=False)

        self.assertEqual(quadratic.get_num_vars(), 4)
        self.assertEqual(quadratic.get_num_linear_constraints(), 0)
        self.assertEqual(quadratic.get_num_quadratic_constraints(), 0)
        self.assertEqual(quadratic.objective.sense, quadratic.objective.Sense.MINIMIZE)
        self.assertAlmostEqual(quadratic.objective.constant, 900000)

        quadratic_matrix = np.zeros((4, 4))
        quadratic_matrix[0, 0] = -500001
        quadratic_matrix[0, 1] = 400000
        quadratic_matrix[0, 2] = 600000
        quadratic_matrix[0, 3] = 800000
        quadratic_matrix[1, 1] = -800001
        quadratic_matrix[1, 2] = 1200000
        quadratic_matrix[1, 3] = 1600000
        quadratic_matrix[2, 2] = -900001
        quadratic_matrix[2, 3] = 2400000
        quadratic_matrix[3, 3] = -800001

        np.testing.assert_array_almost_equal(
            quadratic.objective.quadratic.coefficients.toarray(), quadratic_matrix
        )

    def test_continuous_variable_decode(self):
        """ Test decode func of IntegerToBinaryConverter for continuous variables"""
        try:
            mdl = Model('test_continuous_varable_decode')
            c = mdl.continuous_var(lb=0, ub=10.9, name='c')
            x = mdl.binary_var(name='x')
            mdl.maximize(c + x * x)
            op = QuadraticProgram()
            op.from_docplex(mdl)
            converter = IntegerToBinary()
            op = converter.convert(op)
            admm_params = ADMMParameters()
            qubo_optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
            continuous_optimizer = CplexOptimizer()
            solver = ADMMOptimizer(
                qubo_optimizer=qubo_optimizer,
                continuous_optimizer=continuous_optimizer,
                params=admm_params,
            )
            result = solver.solve(op)
            new_x = converter.interpret(result.x)
            self.assertEqual(new_x[0], 10.9)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_auto_penalty(self):
        """ Test auto penalty function"""
        op = QuadraticProgram()
        op.binary_var('x')
        op.binary_var('y')
        op.binary_var('z')
        op.minimize(constant=3, linear={'x': 1}, quadratic={('x', 'y'): 2})
        op.linear_constraint(linear={'x': 1, 'y': 1, 'z': 1}, sense='EQ', rhs=2, name='xyz_eq')
        lineq2penalty = LinearEqualityToPenalty(penalty=1e5)
        lineq2penalty_auto = LinearEqualityToPenalty()
        qubo = lineq2penalty.convert(op)
        qubo_auto = lineq2penalty_auto.convert(op)
        exact_mes = NumPyMinimumEigensolver()
        exact = MinimumEigenOptimizer(exact_mes)
        result = exact.solve(qubo)
        result_auto = exact.solve(qubo_auto)
        self.assertEqual(result.fval, result_auto.fval)
        np.testing.assert_array_almost_equal(result.x, result_auto.x)

    def test_auto_penalty_warning(self):
        """ Test warnings of auto penalty function"""
        op = QuadraticProgram()
        op.binary_var('x')
        op.binary_var('y')
        op.binary_var('z')
        op.minimize(linear={'x': 1, 'y': 2})
        op.linear_constraint(linear={'x': 0.5, 'y': 0.5, 'z': 0.5}, sense='EQ', rhs=1, name='xyz')
        with self.assertLogs('qiskit.optimization', level='WARNING') as log:
            lineq2penalty = LinearEqualityToPenalty()
            _ = lineq2penalty.convert(op)
        warning = (
            'WARNING:qiskit.optimization.converters.linear_equality_to_penalty:'
            'Warning: Using 100000.000000 for the penalty coefficient because a float '
            'coefficient exists in constraints. \nThe value could be too small. If so, '
            'set the penalty coefficient manually.'
        )
        self.assertIn(warning, log.output)

    def test_linear_equality_to_penalty_decode(self):
        """ Test decode func of LinearEqualityToPenalty"""
        qprog = QuadraticProgram()
        qprog.binary_var('x')
        qprog.binary_var('y')
        qprog.binary_var('z')
        qprog.maximize(linear={'x': 3, 'y': 1, 'z': 1})
        qprog.linear_constraint(linear={'x': 1, 'y': 1, 'z': 1}, sense='EQ', rhs=2, name='xyz_eq')
        lineq2penalty = LinearEqualityToPenalty()
        qubo = lineq2penalty.convert(qprog)
        exact_mes = NumPyMinimumEigensolver()
        exact = MinimumEigenOptimizer(exact_mes)
        result = exact.solve(qubo)

        new_x = lineq2penalty.interpret(result.x)
        np.testing.assert_array_almost_equal(new_x, [1, 1, 0])
        infeasible_x = lineq2penalty.interpret([1, 1, 1])
        np.testing.assert_array_almost_equal(infeasible_x, [1, 1, 1])

    def test_0var_range_inequality(self):
        """ Test InequalityToEquality converter when the var_rang of the slack variable is 0"""
        op = QuadraticProgram()
        op.binary_var('x')
        op.binary_var('y')
        op.linear_constraint(linear={'x': 1, 'y': 1}, sense='LE', rhs=0, name='xy_leq1')
        op.linear_constraint(linear={'x': 1, 'y': 1}, sense='GE', rhs=2, name='xy_geq1')
        op.quadratic_constraint(quadratic={('x', 'x'): 1}, sense='LE', rhs=0, name='xy_leq2')
        op.quadratic_constraint(quadratic={('x', 'y'): 1}, sense='GE', rhs=1, name='xy_geq2')
        ineq2eq = InequalityToEquality()
        new_op = ineq2eq.convert(op)
        self.assertEqual(new_op.get_num_vars(), 2)
        self.assertTrue(all(l_const.sense == Constraint.Sense.EQ
                            for l_const in new_op.linear_constraints))
        self.assertTrue(all(q_const.sense == Constraint.Sense.EQ
                            for q_const in new_op.quadratic_constraints))

    def test_integer_to_binary2(self):
        """Test integer to binary variables 2"""
        mod = QuadraticProgram()
        mod.integer_var(name='x', lowerbound=0, upperbound=1)
        mod.integer_var(name='y', lowerbound=0, upperbound=1)
        mod.minimize(1, {'x': 1}, {('x', 'y'): 2})
        mod.linear_constraint({'x': 1}, '==', 1)
        mod.quadratic_constraint({'x': 1}, {('x', 'y'): 2}, '==', 1)
        mod2 = IntegerToBinary().convert(mod)
        self.assertListEqual([e.name + '@0' for e in mod.variables],
                             [e.name for e in mod2.variables])
        self.assertDictEqual(mod.objective.linear.to_dict(),
                             mod2.objective.linear.to_dict())
        self.assertDictEqual(mod.objective.quadratic.to_dict(),
                             mod2.objective.quadratic.to_dict())
        self.assertEqual(mod.get_num_linear_constraints(),
                         mod2.get_num_linear_constraints())
        for cst, cst2 in zip(mod.linear_constraints, mod2.linear_constraints):
            self.assertDictEqual(cst.linear.to_dict(),
                                 cst2.linear.to_dict())
        self.assertEqual(mod.get_num_quadratic_constraints(),
                         mod2.get_num_quadratic_constraints())
        for cst, cst2 in zip(mod.quadratic_constraints, mod2.quadratic_constraints):
            self.assertDictEqual(cst.linear.to_dict(),
                                 cst2.linear.to_dict())
            self.assertDictEqual(cst.quadratic.to_dict(),
                                 cst2.quadratic.to_dict())


if __name__ == '__main__':
    unittest.main()
