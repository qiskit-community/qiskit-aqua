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

""" Test QuadraticProgram """

import logging
import unittest

from qiskit.optimization import QuadraticProgram, QiskitOptimizationError, infinity
from qiskit.optimization.problems import VarType
from test.optimization.optimization_test_case import QiskitOptimizationTestCase

logger = logging.getLogger(__name__)


class TestQuadraticProgram(QiskitOptimizationTestCase):
    """Test QuadraticProgram without the members that have separate test classes
    (VariablesInterface, etc)."""

    def test_constructor(self):
        """ test constructor """
        quadratic_program = QuadraticProgram()
        self.assertEqual(quadratic_program.name, '')

        quadratic_program = QuadraticProgram('test')
        self.assertEqual(quadratic_program.name, 'test')

        quadratic_program.name = ''
        self.assertEqual(quadratic_program.name, '')

    def test_variables_handling(self):
        """ test add variables """
        quadratic_program = QuadraticProgram()

        self.assertEqual(quadratic_program.get_num_vars(), 0)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 0)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 0)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_0 = quadratic_program.continuous_var()
        self.assertEqual(x_0.name, 'x0')
        self.assertEqual(x_0.lowerbound, 0)
        self.assertEqual(x_0.upperbound, infinity)
        self.assertEqual(x_0.vartype, VarType.CONTINUOUS)

        self.assertEqual(quadratic_program.get_num_vars(), 1)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 1)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 0)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_1 = quadratic_program.continuous_var(name='x1', lowerbound=5, upperbound=10)
        self.assertEqual(x_1.name, 'x1')
        self.assertEqual(x_1.lowerbound, 5)
        self.assertEqual(x_1.upperbound, 10)
        self.assertEqual(x_1.vartype, VarType.CONTINUOUS)

        self.assertEqual(quadratic_program.get_num_vars(), 2)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 0)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_2 = quadratic_program.binary_var()
        self.assertEqual(x_2.name, 'x2')
        self.assertEqual(x_2.lowerbound, 0)
        self.assertEqual(x_2.upperbound, 1)
        self.assertEqual(x_2.vartype, VarType.BINARY)

        self.assertEqual(quadratic_program.get_num_vars(), 3)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 1)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_3 = quadratic_program.binary_var(name='x3')
        self.assertEqual(x_3.name, 'x3')
        self.assertEqual(x_3.lowerbound, 0)
        self.assertEqual(x_3.upperbound, 1)
        self.assertEqual(x_3.vartype, VarType.BINARY)

        self.assertEqual(quadratic_program.get_num_vars(), 4)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 2)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_4 = quadratic_program.integer_var()
        self.assertEqual(x_4.name, 'x4')
        self.assertEqual(x_4.lowerbound, 0)
        self.assertEqual(x_4.upperbound, infinity)
        self.assertEqual(x_4.vartype, VarType.INTEGER)

        self.assertEqual(quadratic_program.get_num_vars(), 5)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 2)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 1)

        x_5 = quadratic_program.integer_var(name='x5', lowerbound=5, upperbound=10)
        self.assertEqual(x_5.name, 'x5')
        self.assertEqual(x_5.lowerbound, 5)
        self.assertEqual(x_5.upperbound, 10)
        self.assertEqual(x_5.vartype, VarType.INTEGER)

        self.assertEqual(quadratic_program.get_num_vars(), 6)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 2)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 2)

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.continuous_var(name='x0')

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.binary_var(name='x0')

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.integer_var(name='x0')

        variables = [x_0, x_1, x_2, x_3, x_4, x_5]
        for i, x in enumerate(variables):
            y = quadratic_program.get_variable(i)
            z = quadratic_program.get_variable(x.name)
            self.assertEqual(x, y)
            self.assertEqual(x, z)

    def test_linear_constraints_handling(self):
        # TODO
        pass

    def test_quadratic_constraints_handling(self):
        # TODO
        pass

    def test_objective_handling(self):
        # TODO
        pass

    def test_read_problem(self):
        # TODO
        pass

    def test_write_problem(self):
        # TODO
        pass

    def test_docplex(self):
        q_p = QuadraticProgram('test')
        q_p.binary_var('x')
        q_p.integer_var('y', lowerbound=-2, upperbound=4)
        q_p.continuous_var('z', lowerbound=-1.5, upperbound=3.2)
        q_p.minimize(constant=1, linear={'x': 1, 'y': 2},
                     quadratic={('x', 'y'): -1, ('z', 'z'): 2})
        q_p.linear_constraint({'x': 2, 'z': -1}, '==', 1)
        q_p.quadratic_constraint({'x': 2, 'z': -1}, {('y', 'z'): 3}, '==', 1)
        q_p2 = QuadraticProgram()
        q_p2.from_docplex(q_p.to_docplex())
        self.assertEqual(q_p.pprint_as_string(), q_p2.pprint_as_string())
        self.assertEqual(q_p.print_as_lp_string(), q_p2.print_as_lp_string())

    def test_substitute_variables(self):
        q_p = QuadraticProgram('test')
        q_p.binary_var('x')
        q_p.integer_var('y', lowerbound=-2, upperbound=4)
        q_p.continuous_var('z', lowerbound=-1.5, upperbound=3.2)
        q_p.minimize(constant=1, linear={'x': 1, 'y': 2},
                     quadratic={('x', 'y'): -1, ('z', 'z'): 2})
        q_p.linear_constraint({'x': 2, 'z': -1}, '==', 1)
        q_p.quadratic_constraint({'x': 2, 'z': -1}, {('y', 'z'): 3}, '<=', -1)
        print(q_p.print_as_lp_string())

        q_p2, status = q_p.substitute_variables(constants={'x': -1})
        self.assertEqual(status.name, 'infeasible')
        q_p2, status = q_p.substitute_variables(constants={'y': -3})
        self.assertEqual(status.name, 'infeasible')

        q_p2, status = q_p.substitute_variables(constants={'x': 0})
        self.assertDictEqual(q_p2.objective.linear.to_dict(use_index=False), {'y': 2})
        self.assertDictEqual(q_p2.objective.quadratic.to_dict(use_index=False), {('z', 'z'): 2})
        self.assertEqual(q_p2.objective.constant, 1)
        self.assertEqual(len(q_p2.linear_constraints), 1)
        self.assertEqual(len(q_p2.quadratic_constraints), 1)

        cst = q_p2.linear_constraints[0]
        self.assertDictEqual(cst.linear.to_dict(use_index=False), {'z': -1})
        self.assertEqual(cst.sense.name, 'EQ')
        self.assertEqual(cst.rhs, 1)

        cst = q_p2.quadratic_constraints[0]
        self.assertDictEqual(cst.linear.to_dict(use_index=False), {'z': -1})
        self.assertDictEqual(cst.quadratic.to_dict(use_index=False), {('y', 'z'): 3})
        self.assertEqual(cst.sense.name, 'LE')
        self.assertEqual(cst.rhs, -1)

        q_p2, status = q_p.substitute_variables(constants={'z': -1})
        self.assertDictEqual(q_p2.objective.linear.to_dict(use_index=False), {'x': 1, 'y': 2})
        self.assertDictEqual(q_p2.objective.quadratic.to_dict(use_index=False), {('x', 'y'): -1})
        self.assertEqual(q_p2.objective.constant, 3)
        self.assertEqual(len(q_p2.linear_constraints), 2)
        self.assertEqual(len(q_p2.quadratic_constraints), 0)

        cst = q_p2.linear_constraints[0]
        self.assertDictEqual(cst.linear.to_dict(use_index=False), {'x': 2})
        self.assertEqual(cst.sense.name, 'EQ')
        self.assertEqual(cst.rhs, 0)

        cst = q_p2.linear_constraints[1]
        self.assertDictEqual(cst.linear.to_dict(use_index=False), {'x': 2, 'y': -3})
        self.assertEqual(cst.sense.name, 'LE')
        self.assertEqual(cst.rhs, -2)

        q_p2, status = q_p.substitute_variables(variables={'y': ('x', -0.5)})
        print(q_p2.print_as_lp_string())
        self.assertDictEqual(q_p2.objective.linear.to_dict(use_index=False), {})
        self.assertDictEqual(q_p2.objective.quadratic.to_dict(use_index=False),
                             {('x', 'x'): 0.5, ('z', 'z'): 2})
        self.assertEqual(q_p2.objective.constant, 1)
        self.assertEqual(len(q_p2.linear_constraints), 1)
        self.assertEqual(len(q_p2.quadratic_constraints), 1)

        cst = q_p2.linear_constraints[0]
        self.assertDictEqual(cst.linear.to_dict(use_index=False), {'x': 2, 'z': -1})
        self.assertEqual(cst.sense.name, 'EQ')
        self.assertEqual(cst.rhs, 1)

        cst = q_p2.quadratic_constraints[0]
        self.assertDictEqual(cst.linear.to_dict(use_index=False), {'x': 2, 'z': -1})
        self.assertDictEqual(cst.quadratic.to_dict(use_index=False), {('x', 'z'): -1.5})
        self.assertEqual(cst.sense.name, 'LE')
        self.assertEqual(cst.rhs, -1)


if __name__ == '__main__':
    unittest.main()
