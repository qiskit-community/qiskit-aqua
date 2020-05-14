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
import numpy as np
from docplex.mp.model import Model

from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.optimization import QuadraticProgram, QiskitOptimizationError
from qiskit.optimization.problems import Constraint, Variable
from qiskit.optimization.algorithms import OptimizationResult
from qiskit.optimization.converters import (
    InequalityToEquality,
    QuadraticProgramToIsing,
    IsingToQuadraticProgram,
    IntegerToBinary,
    LinearEqualityToPenalty,
)
from qiskit.optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer, ADMMOptimizer
from qiskit.optimization.algorithms.admm_optimizer import ADMMParameters
from qiskit.quantum_info import Pauli

logger = logging.getLogger(__name__)


QUBIT_OP_MAXIMIZE_SAMPLE = WeightedPauliOperator(
    paulis=[
        [(-199999.5 + 0j), Pauli(z=[True, False, False, False], x=[False, False, False, False])],
        [(-399999.5 + 0j), Pauli(z=[False, True, False, False], x=[False, False, False, False])],
        [(-599999.5 + 0j), Pauli(z=[False, False, True, False], x=[False, False, False, False])],
        [(-799999.5 + 0j), Pauli(z=[False, False, False, True], x=[False, False, False, False])],
        [(100000 + 0j), Pauli(z=[True, True, False, False], x=[False, False, False, False])],
        [(150000 + 0j), Pauli(z=[True, False, True, False], x=[False, False, False, False])],
        [(300000 + 0j), Pauli(z=[False, True, True, False], x=[False, False, False, False])],
        [(200000 + 0j), Pauli(z=[True, False, False, True], x=[False, False, False, False])],
        [(400000 + 0j), Pauli(z=[False, True, False, True], x=[False, False, False, False])],
        [(600000 + 0j), Pauli(z=[False, False, True, True], x=[False, False, False, False])],
    ]
)
OFFSET_MAXIMIZE_SAMPLE = 1149998


class TestConverters(QiskitOptimizationTestCase):
    """Test Converters"""

    def test_empty_problem(self):
        """ Test empty problem """
        op = QuadraticProgram()
        conv = InequalityToEquality()
        op = conv.encode(op)
        conv = IntegerToBinary()
        op = conv.encode(op)
        conv = LinearEqualityToPenalty()
        op = conv.encode(op)
        conv = QuadraticProgramToIsing()
        _, shift = conv.encode(op)
        self.assertEqual(shift, 0.0)

    def test_valid_variable_type(self):
        """Validate the types of the variables for QuadraticProgramToIsing."""
        # Integer variable
        with self.assertRaises(QiskitOptimizationError):
            op = QuadraticProgram()
            op.integer_var(0, 10, "int_var")
            conv = QuadraticProgramToIsing()
            _ = conv.encode(op)
        # Continuous variable
        with self.assertRaises(QiskitOptimizationError):
            op = QuadraticProgram()
            op.continuous_var(0, 10, "continuous_var")
            conv = QuadraticProgramToIsing()
            _ = conv.encode(op)

    def test_inequality_binary(self):
        """ Test InequalityToEqualityConverter with binary variables """
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name='x{}'.format(i))
        # Linear constraints
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x1'] = 1
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {}
        linear_constraint['x1'] = 1
        linear_constraint['x2'] = -1
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, 'x1x2')
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x2'] = 3
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 2, 'x0x2')
        # Quadratic constraints
        quadratic = {}
        quadratic[('x0', 'x1')] = 1
        quadratic[('x1', 'x2')] = 2
        op.quadratic_constraint({}, quadratic, Constraint.Sense.LE, 3, 'x0x1_x1x2LE')
        quadratic = {}
        quadratic[('x0', 'x1')] = 3
        quadratic[('x1', 'x2')] = 4
        op.quadratic_constraint({}, quadratic, Constraint.Sense.GE, 3, 'x0x1_x1x2GE')
        # Convert inequality constraints into equality constraints
        conv = InequalityToEquality()
        op2 = conv.encode(op)
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

    def test_inequality_integer(self):
        """ Test InequalityToEqualityConverter with integer variables """
        op = QuadraticProgram()
        for i in range(3):
            op.integer_var(name='x{}'.format(i), lowerbound=-3, upperbound=3)
        # Linear constraints
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x1'] = 1
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {}
        linear_constraint['x1'] = 1
        linear_constraint['x2'] = -1
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, 'x1x2')
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x2'] = 3
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 2, 'x0x2')
        # Quadratic constraints
        quadratic = {}
        quadratic[('x0', 'x1')] = 1
        quadratic[('x1', 'x2')] = 2
        op.quadratic_constraint({}, quadratic, Constraint.Sense.LE, 3, 'x0x1_x1x2LE')
        quadratic = {}
        quadratic[('x0', 'x1')] = 3
        quadratic[('x1', 'x2')] = 4
        op.quadratic_constraint({}, quadratic, Constraint.Sense.GE, 3, 'x0x1_x1x2GE')
        conv = InequalityToEquality()
        op2 = conv.encode(op)
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

    def test_inequality_mode_integer(self):
        """ Test integer mode of InequalityToEqualityConverter() """
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name='x{}'.format(i))
        # Linear constraints
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x1'] = 1
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {}
        linear_constraint['x1'] = 1
        linear_constraint['x2'] = -1
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, 'x1x2')
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x2'] = 3
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 2, 'x0x2')
        conv = InequalityToEquality()
        op2 = conv.encode(op, mode='integer')
        lst = [op2.variables[3].vartype, op2.variables[4].vartype]
        self.assertListEqual(lst, [Variable.Type.INTEGER, Variable.Type.INTEGER])

    def test_inequality_mode_continuous(self):
        """ Test continuous mode of InequalityToEqualityConverter() """
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name='x{}'.format(i))
        # Linear constraints
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x1'] = 1
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {}
        linear_constraint['x1'] = 1
        linear_constraint['x2'] = -1
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, 'x1x2')
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x2'] = 3
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 2, 'x0x2')
        conv = InequalityToEquality()
        op2 = conv.encode(op, mode='continuous')
        lst = [op2.variables[3].vartype, op2.variables[4].vartype]
        self.assertListEqual(lst, [Variable.Type.CONTINUOUS, Variable.Type.CONTINUOUS])

    def test_inequality_mode_auto(self):
        """ Test auto mode of InequalityToEqualityConverter() """
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name='x{}'.format(i))
        # Linear constraints
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x1'] = 1
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {}
        linear_constraint['x1'] = 1
        linear_constraint['x2'] = -1
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, 'x1x2')
        linear_constraint = {}
        linear_constraint['x0'] = 1.1
        linear_constraint['x2'] = 2.2
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 3.3, 'x0x2')
        conv = InequalityToEquality()
        op2 = conv.encode(op, mode='auto')
        lst = [op2.variables[3].vartype, op2.variables[4].vartype]
        self.assertListEqual(lst, [Variable.Type.INTEGER, Variable.Type.CONTINUOUS])

    def test_penalize_sense(self):
        """ Test PenalizeLinearEqualityConstraints with senses """
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name='x{}'.format(i))
        # Linear constraints
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x1'] = 1
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {}
        linear_constraint['x1'] = 1
        linear_constraint['x2'] = -1
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, 'x1x2')
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x2'] = 3
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 2, 'x0x2')
        self.assertEqual(len(op.linear_constraints), 3)
        conv = LinearEqualityToPenalty()
        with self.assertRaises(QiskitOptimizationError):
            conv.encode(op)

    def test_penalize_binary(self):
        """ Test PenalizeLinearEqualityConstraints with binary variables """
        op = QuadraticProgram()
        for i in range(3):
            op.binary_var(name='x{}'.format(i))
        # Linear constraints
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x1'] = 1
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {}
        linear_constraint['x1'] = 1
        linear_constraint['x2'] = -1
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 2, 'x1x2')
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x2'] = 3
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 2, 'x0x2')
        self.assertEqual(len(op.linear_constraints), 3)
        conv = LinearEqualityToPenalty()
        op2 = conv.encode(op)
        self.assertEqual(len(op2.linear_constraints), 0)

    def test_penalize_integer(self):
        """ Test PenalizeLinearEqualityConstraints with integer variables """
        op = QuadraticProgram()
        for i in range(3):
            op.integer_var(name='x{}'.format(i), lowerbound=-3, upperbound=3)
        # Linear constraints
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x1'] = 1
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, 'x0x1')
        linear_constraint = {}
        linear_constraint['x1'] = 1
        linear_constraint['x2'] = -1
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 2, 'x1x2')
        linear_constraint = {}
        linear_constraint['x0'] = 1
        linear_constraint['x2'] = 3
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 2, 'x0x2')
        self.assertEqual(len(op.linear_constraints), 3)
        conv = LinearEqualityToPenalty()
        op2 = conv.encode(op)
        self.assertEqual(len(op2.linear_constraints), 0)

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
        op2 = conv.encode(op)
        for x in op2.variables:
            self.assertEqual(x.vartype, Variable.Type.BINARY)
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
        linear = {}
        linear['x0'] = 1
        linear['x1'] = 2
        linear['x2'] = 1
        op.maximize(0, linear, {})
        linear = {}
        for x in op.variables:
            linear[x.name] = 1
        op.linear_constraint(linear, Constraint.Sense.EQ, 6, 'x0x1x2')
        conv = IntegerToBinary()
        _ = conv.encode(op)
        result = OptimizationResult(x=[0, 1, 1, 1, 1], fval=17)
        new_result = conv.decode(result)
        self.assertListEqual(new_result.x, [0, 1, 5])
        self.assertEqual(new_result.fval, 17)

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
        penalize = LinearEqualityToPenalty()
        op2ope = QuadraticProgramToIsing()
        op2 = penalize.encode(op)
        qubitop, offset = op2ope.encode(op2)

        # the encoder uses a dictionary, in which the order of items in Python 3.5 is not
        # maintained, therefore don't do a list compare but dictionary compare
        qubit_op_as_dict = dict(qubitop.paulis)
        for coeff, paulis in QUBIT_OP_MAXIMIZE_SAMPLE.paulis:
            self.assertEqual(paulis, qubit_op_as_dict[coeff])

        self.assertEqual(offset, OFFSET_MAXIMIZE_SAMPLE)

    def test_ising_to_quadraticprogram_linear(self):
        """ Test optimization problem to operators with linear=True"""
        op = QUBIT_OP_MAXIMIZE_SAMPLE
        offset = OFFSET_MAXIMIZE_SAMPLE

        op2qp = IsingToQuadraticProgram(linear=True)
        quadratic = op2qp.encode(op, offset)

        self.assertEqual(len(quadratic.variables), 4)
        self.assertEqual(len(quadratic.linear_constraints), 0)
        self.assertEqual(len(quadratic.quadratic_constraints), 0)
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

        np.testing.assert_array_almost_equal(quadratic.objective.linear.coefficients.toarray(),
                                             linear_matrix)
        np.testing.assert_array_almost_equal(quadratic.objective.quadratic.coefficients.toarray(),
                                             quadratic_matrix)

    def test_ising_to_quadraticprogram_quadratic(self):
        """ Test optimization problem to operators with linear=False"""
        op = QUBIT_OP_MAXIMIZE_SAMPLE
        offset = OFFSET_MAXIMIZE_SAMPLE

        op2qp = IsingToQuadraticProgram(linear=False)
        quadratic = op2qp.encode(op, offset)

        self.assertEqual(len(quadratic.variables), 4)
        self.assertEqual(len(quadratic.linear_constraints), 0)
        self.assertEqual(len(quadratic.quadratic_constraints), 0)
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

        np.testing.assert_array_almost_equal(quadratic.objective.quadratic.coefficients.toarray(),
                                             quadratic_matrix)

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
            op = converter.encode(op)
            admm_params = ADMMParameters()
            qubo_optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
            continuous_optimizer = CplexOptimizer()
            solver = ADMMOptimizer(
                qubo_optimizer=qubo_optimizer,
                continuous_optimizer=continuous_optimizer,
                params=admm_params,
            )
            solution = solver.solve(op)
            solution = converter.decode(solution)
            self.assertEqual(solution.x[0], 10.9)
        except NameError as ex:
            self.skipTest(str(ex))


if __name__ == '__main__':
    unittest.main()
