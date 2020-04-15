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

import unittest
from test.optimization.optimization_test_case import QiskitOptimizationTestCase
import logging

from qiskit.optimization import QuadraticProgram, QiskitOptimizationError, infinity
from qiskit.optimization.problems import VarType

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
        self.assertEqual(x_0.vartype, VarType.continuous)

        self.assertEqual(quadratic_program.get_num_vars(), 1)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 1)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 0)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_1 = quadratic_program.continuous_var(name='x1', lowerbound=5, upperbound=10)
        self.assertEqual(x_1.name, 'x1')
        self.assertEqual(x_1.lowerbound, 5)
        self.assertEqual(x_1.upperbound, 10)
        self.assertEqual(x_1.vartype, VarType.continuous)

        self.assertEqual(quadratic_program.get_num_vars(), 2)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 0)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_2 = quadratic_program.binary_var()
        self.assertEqual(x_2.name, 'x2')
        self.assertEqual(x_2.lowerbound, 0)
        self.assertEqual(x_2.upperbound, 1)
        self.assertEqual(x_2.vartype, VarType.binary)

        self.assertEqual(quadratic_program.get_num_vars(), 3)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 1)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_3 = quadratic_program.binary_var(name='x3')
        self.assertEqual(x_3.name, 'x3')
        self.assertEqual(x_3.lowerbound, 0)
        self.assertEqual(x_3.upperbound, 1)
        self.assertEqual(x_3.vartype, VarType.binary)

        self.assertEqual(quadratic_program.get_num_vars(), 4)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 2)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_4 = quadratic_program.integer_var()
        self.assertEqual(x_4.name, 'x4')
        self.assertEqual(x_4.lowerbound, 0)
        self.assertEqual(x_4.upperbound, infinity)
        self.assertEqual(x_4.vartype, VarType.integer)

        self.assertEqual(quadratic_program.get_num_vars(), 5)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 2)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 1)

        x_5 = quadratic_program.integer_var(name='x5', lowerbound=5, upperbound=10)
        self.assertEqual(x_5.name, 'x5')
        self.assertEqual(x_5.lowerbound, 5)
        self.assertEqual(x_5.upperbound, 10)
        self.assertEqual(x_5.vartype, VarType.integer)

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


if __name__ == '__main__':
    unittest.main()
