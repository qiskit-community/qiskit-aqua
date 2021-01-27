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

""" Test LinearConstraint """

import unittest
from test.optimization.optimization_test_case import QiskitOptimizationTestCase

import numpy as np

from qiskit.optimization import QuadraticProgram, QiskitOptimizationError
from qiskit.optimization.problems import Constraint


class TestLinearConstraint(QiskitOptimizationTestCase):
    """Test LinearConstraint."""

    def test_init(self):
        """ test init. """

        quadratic_program = QuadraticProgram()
        for _ in range(5):
            quadratic_program.continuous_var()
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 0)

        coefficients = np.array(range(5))

        # equality constraints
        quadratic_program.linear_constraint(sense='==')
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 1)
        self.assertEqual(quadratic_program.linear_constraints[0].name, 'c0')
        self.assertEqual(len(quadratic_program.linear_constraints[0].linear.to_dict()), 0)
        self.assertEqual(quadratic_program.linear_constraints[0].sense, Constraint.Sense.EQ)
        self.assertEqual(quadratic_program.linear_constraints[0].rhs, 0.0)
        self.assertEqual(quadratic_program.linear_constraints[0],
                         quadratic_program.get_linear_constraint('c0'))
        self.assertEqual(quadratic_program.linear_constraints[0],
                         quadratic_program.get_linear_constraint(0))

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.linear_constraint(name='c0')

        quadratic_program.linear_constraint(coefficients, '==', 1.0, 'c1')
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 2)
        self.assertEqual(quadratic_program.linear_constraints[1].name, 'c1')
        self.assertTrue(
            (quadratic_program.linear_constraints[1].linear.to_array() == coefficients).all())
        self.assertEqual(quadratic_program.linear_constraints[1].sense, Constraint.Sense.EQ)
        self.assertEqual(quadratic_program.linear_constraints[1].rhs, 1.0)
        self.assertEqual(quadratic_program.linear_constraints[1],
                         quadratic_program.get_linear_constraint('c1'))
        self.assertEqual(quadratic_program.linear_constraints[1],
                         quadratic_program.get_linear_constraint(1))

        # geq constraints
        quadratic_program.linear_constraint(sense='>=')
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 3)
        self.assertEqual(quadratic_program.linear_constraints[2].name, 'c2')
        self.assertEqual(len(quadratic_program.linear_constraints[2].linear.to_dict()),
                         0)
        self.assertEqual(quadratic_program.linear_constraints[2].sense, Constraint.Sense.GE)
        self.assertEqual(quadratic_program.linear_constraints[2].rhs, 0.0)
        self.assertEqual(quadratic_program.linear_constraints[2],
                         quadratic_program.get_linear_constraint('c2'))
        self.assertEqual(quadratic_program.linear_constraints[2],
                         quadratic_program.get_linear_constraint(2))

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.linear_constraint(name='c2', sense='>=')

        quadratic_program.linear_constraint(coefficients, '>=', 1.0, 'c3')
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 4)
        self.assertEqual(quadratic_program.linear_constraints[3].name, 'c3')
        self.assertTrue(
            (quadratic_program.linear_constraints[3].linear.to_array() == coefficients).all())
        self.assertEqual(quadratic_program.linear_constraints[3].sense, Constraint.Sense.GE)
        self.assertEqual(quadratic_program.linear_constraints[3].rhs, 1.0)
        self.assertEqual(quadratic_program.linear_constraints[3],
                         quadratic_program.get_linear_constraint('c3'))
        self.assertEqual(quadratic_program.linear_constraints[3],
                         quadratic_program.get_linear_constraint(3))

        # leq constraints
        quadratic_program.linear_constraint(sense='<=')
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 5)
        self.assertEqual(quadratic_program.linear_constraints[4].name, 'c4')
        self.assertEqual(len(quadratic_program.linear_constraints[4].linear.to_dict()), 0)
        self.assertEqual(quadratic_program.linear_constraints[4].sense, Constraint.Sense.LE)
        self.assertEqual(quadratic_program.linear_constraints[4].rhs, 0.0)
        self.assertEqual(quadratic_program.linear_constraints[4],
                         quadratic_program.get_linear_constraint('c4'))
        self.assertEqual(quadratic_program.linear_constraints[4],
                         quadratic_program.get_linear_constraint(4))

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.linear_constraint(name='c4', sense='<=')

        quadratic_program.linear_constraint(coefficients, '<=', 1.0, 'c5')
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 6)
        self.assertEqual(quadratic_program.linear_constraints[5].name, 'c5')
        self.assertTrue(
            (quadratic_program.linear_constraints[5].linear.to_array() == coefficients).all())
        self.assertEqual(quadratic_program.linear_constraints[5].sense, Constraint.Sense.LE)
        self.assertEqual(quadratic_program.linear_constraints[5].rhs, 1.0)
        self.assertEqual(quadratic_program.linear_constraints[5],
                         quadratic_program.get_linear_constraint('c5'))
        self.assertEqual(quadratic_program.linear_constraints[5],
                         quadratic_program.get_linear_constraint(5))


if __name__ == '__main__':
    unittest.main()
