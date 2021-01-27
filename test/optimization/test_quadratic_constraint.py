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

""" Test QuadraticConstraint """

import unittest
from test.optimization.optimization_test_case import QiskitOptimizationTestCase

import numpy as np

from qiskit.optimization import QuadraticProgram, QiskitOptimizationError
from qiskit.optimization.problems import Constraint


class TestQuadraticConstraint(QiskitOptimizationTestCase):
    """Test QuadraticConstraint."""

    def test_init(self):
        """ test init. """

        quadratic_program = QuadraticProgram()
        for _ in range(5):
            quadratic_program.continuous_var()
        self.assertEqual(quadratic_program.get_num_quadratic_constraints(), 0)

        linear_coeffs = np.array(range(5))
        lst = [[0 for _ in range(5)] for _ in range(5)]
        for i, v in enumerate(lst):
            for j, _ in enumerate(v):
                lst[min(i, j)][max(i, j)] += i * j
        quadratic_coeffs = np.array(lst)

        # equality constraints
        quadratic_program.quadratic_constraint(sense='==')
        self.assertEqual(quadratic_program.get_num_quadratic_constraints(), 1)
        self.assertEqual(quadratic_program.quadratic_constraints[0].name, 'q0')
        self.assertEqual(len(quadratic_program.quadratic_constraints[0].linear.to_dict()), 0)
        self.assertEqual(len(quadratic_program.quadratic_constraints[0].quadratic.to_dict()), 0)
        self.assertEqual(quadratic_program.quadratic_constraints[0].sense, Constraint.Sense.EQ)
        self.assertEqual(quadratic_program.quadratic_constraints[0].rhs, 0.0)
        self.assertEqual(quadratic_program.quadratic_constraints[0],
                         quadratic_program.get_quadratic_constraint('q0'))
        self.assertEqual(quadratic_program.quadratic_constraints[0],
                         quadratic_program.get_quadratic_constraint(0))

        self.assertEqual(quadratic_program.quadratic_constraints[0].evaluate(linear_coeffs), 0.0)

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.quadratic_constraint(name='q0', sense='==')

        quadratic_program.quadratic_constraint(linear_coeffs, quadratic_coeffs, '==', 1.0, 'q1')
        self.assertEqual(quadratic_program.get_num_quadratic_constraints(), 2)
        self.assertEqual(quadratic_program.quadratic_constraints[1].name, 'q1')
        self.assertTrue(
            (quadratic_program.quadratic_constraints[1].linear.to_array() == linear_coeffs).all())
        self.assertTrue(
            (quadratic_program.quadratic_constraints[1].quadratic.to_array() ==
             quadratic_coeffs).all())
        self.assertEqual(quadratic_program.quadratic_constraints[1].sense, Constraint.Sense.EQ)
        self.assertEqual(quadratic_program.quadratic_constraints[1].rhs, 1.0)
        self.assertEqual(quadratic_program.quadratic_constraints[1],
                         quadratic_program.get_quadratic_constraint('q1'))
        self.assertEqual(quadratic_program.quadratic_constraints[1],
                         quadratic_program.get_quadratic_constraint(1))

        self.assertEqual(quadratic_program.quadratic_constraints[1].evaluate(linear_coeffs), 930.0)

        # geq constraints
        quadratic_program.quadratic_constraint(sense='>=')
        self.assertEqual(quadratic_program.get_num_quadratic_constraints(), 3)
        self.assertEqual(quadratic_program.quadratic_constraints[2].name, 'q2')
        self.assertEqual(len(quadratic_program.quadratic_constraints[2].linear.to_dict()), 0)
        self.assertEqual(len(quadratic_program.quadratic_constraints[2].quadratic.to_dict()), 0)
        self.assertEqual(quadratic_program.quadratic_constraints[2].sense, Constraint.Sense.GE)
        self.assertEqual(quadratic_program.quadratic_constraints[2].rhs, 0.0)
        self.assertEqual(quadratic_program.quadratic_constraints[2],
                         quadratic_program.get_quadratic_constraint('q2'))
        self.assertEqual(quadratic_program.quadratic_constraints[2],
                         quadratic_program.get_quadratic_constraint(2))

        self.assertEqual(quadratic_program.quadratic_constraints[2].evaluate(linear_coeffs), 0.0)

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.quadratic_constraint(name='q2', sense='>=')

        quadratic_program.quadratic_constraint(linear_coeffs, quadratic_coeffs, '>=', 1.0, 'q3')
        self.assertEqual(quadratic_program.get_num_quadratic_constraints(), 4)
        self.assertEqual(quadratic_program.quadratic_constraints[3].name, 'q3')
        self.assertTrue(
            (quadratic_program.quadratic_constraints[3].linear.to_array() == linear_coeffs).all())
        self.assertTrue((quadratic_program.quadratic_constraints[3].quadratic.to_array() ==
                         quadratic_coeffs).all())
        self.assertEqual(quadratic_program.quadratic_constraints[3].sense, Constraint.Sense.GE)
        self.assertEqual(quadratic_program.quadratic_constraints[3].rhs, 1.0)
        self.assertEqual(quadratic_program.quadratic_constraints[3],
                         quadratic_program.get_quadratic_constraint('q3'))
        self.assertEqual(quadratic_program.quadratic_constraints[3],
                         quadratic_program.get_quadratic_constraint(3))

        self.assertEqual(quadratic_program.quadratic_constraints[3].evaluate(linear_coeffs), 930.0)

        # leq constraints
        quadratic_program.quadratic_constraint(sense='<=')
        self.assertEqual(quadratic_program.get_num_quadratic_constraints(), 5)
        self.assertEqual(quadratic_program.quadratic_constraints[4].name, 'q4')
        self.assertEqual(len(quadratic_program.quadratic_constraints[4].linear.to_dict()), 0)
        self.assertEqual(quadratic_program.quadratic_constraints[4].sense, Constraint.Sense.LE)
        self.assertEqual(quadratic_program.quadratic_constraints[4].rhs, 0.0)
        self.assertEqual(quadratic_program.quadratic_constraints[4],
                         quadratic_program.get_quadratic_constraint('q4'))
        self.assertEqual(quadratic_program.quadratic_constraints[4],
                         quadratic_program.get_quadratic_constraint(4))

        self.assertEqual(quadratic_program.quadratic_constraints[4].evaluate(linear_coeffs), 0.0)

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.quadratic_constraint(name='q4', sense='<=')

        quadratic_program.quadratic_constraint(linear_coeffs, quadratic_coeffs, '<=', 1.0, 'q5')
        self.assertEqual(quadratic_program.get_num_quadratic_constraints(), 6)
        self.assertEqual(quadratic_program.quadratic_constraints[5].name, 'q5')
        self.assertTrue(
            (quadratic_program.quadratic_constraints[5].linear.to_array() == linear_coeffs).all())
        self.assertTrue((quadratic_program.quadratic_constraints[5].quadratic.to_array() ==
                         quadratic_coeffs).all())
        self.assertEqual(quadratic_program.quadratic_constraints[5].sense, Constraint.Sense.LE)
        self.assertEqual(quadratic_program.quadratic_constraints[5].rhs, 1.0)
        self.assertEqual(quadratic_program.quadratic_constraints[5],
                         quadratic_program.get_quadratic_constraint('q5'))
        self.assertEqual(quadratic_program.quadratic_constraints[5],
                         quadratic_program.get_quadratic_constraint(5))

        self.assertEqual(quadratic_program.quadratic_constraints[5].evaluate(linear_coeffs), 930.0)


if __name__ == '__main__':
    unittest.main()
