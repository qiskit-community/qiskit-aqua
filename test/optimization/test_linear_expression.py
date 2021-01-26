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

""" Test LinearExpression """

import unittest

from test.optimization.optimization_test_case import QiskitOptimizationTestCase
import numpy as np
from scipy.sparse import dok_matrix

from qiskit.optimization import QuadraticProgram
from qiskit.optimization.problems import LinearExpression


class TestLinearExpression(QiskitOptimizationTestCase):
    """Test LinearExpression."""

    def test_init(self):
        """ test init. """

        quadratic_program = QuadraticProgram()
        for _ in range(5):
            quadratic_program.continuous_var()

        coefficients_list = list(range(5))
        coefficients_array = np.array(coefficients_list)
        coefficients_dok = dok_matrix([coefficients_list])
        coefficients_dict_int = {i: i for i in range(1, 5)}
        coefficients_dict_str = {'x{}'.format(i): i for i in range(1, 5)}

        for coeffs in [coefficients_list,
                       coefficients_array,
                       coefficients_dok,
                       coefficients_dict_int,
                       coefficients_dict_str]:
            linear = LinearExpression(quadratic_program, coeffs)
            self.assertEqual((linear.coefficients != coefficients_dok).nnz, 0)
            self.assertTrue((linear.to_array() == coefficients_list).all())
            self.assertDictEqual(linear.to_dict(use_name=False), coefficients_dict_int)
            self.assertDictEqual(linear.to_dict(use_name=True), coefficients_dict_str)

    def test_get_item(self):
        """ test get_item. """

        quadratic_program = QuadraticProgram()
        for _ in range(5):
            quadratic_program.continuous_var()

        coefficients = list(range(5))
        linear = LinearExpression(quadratic_program, coefficients)
        for i, v in enumerate(coefficients):
            self.assertEqual(linear[i], v)

    def test_setters(self):
        """ test setters. """

        quadratic_program = QuadraticProgram()
        for _ in range(5):
            quadratic_program.continuous_var()

        zeros = np.zeros(quadratic_program.get_num_vars())
        linear = LinearExpression(quadratic_program, zeros)

        coefficients_list = list(range(5))
        coefficients_array = np.array(coefficients_list)
        coefficients_dok = dok_matrix([coefficients_list])
        coefficients_dict_int = {i: i for i in range(1, 5)}
        coefficients_dict_str = {'x{}'.format(i): i for i in range(1, 5)}

        for coeffs in [coefficients_list,
                       coefficients_array,
                       coefficients_dok,
                       coefficients_dict_int,
                       coefficients_dict_str]:
            linear.coefficients = coeffs
            self.assertEqual((linear.coefficients != coefficients_dok).nnz, 0)
            self.assertTrue((linear.to_array() == coefficients_list).all())
            self.assertDictEqual(linear.to_dict(use_name=False), coefficients_dict_int)
            self.assertDictEqual(linear.to_dict(use_name=True), coefficients_dict_str)

    def test_evaluate(self):
        """ test evaluate. """

        quadratic_program = QuadraticProgram()
        x = [quadratic_program.continuous_var() for _ in range(5)]

        coefficients_list = list(range(5))
        linear = LinearExpression(quadratic_program, coefficients_list)

        values_list = list(range(len(x)))
        values_array = np.array(values_list)
        values_dict_int = {i: i for i in range(len(x))}
        values_dict_str = {'x{}'.format(i): i for i in range(len(x))}

        for values in [values_list, values_array, values_dict_int, values_dict_str]:
            self.assertEqual(linear.evaluate(values), 30)

    def test_evaluate_gradient(self):
        """ test evaluate gradient. """

        quadratic_program = QuadraticProgram()
        x = [quadratic_program.continuous_var() for _ in range(5)]

        coefficients_list = list(range(5))
        linear = LinearExpression(quadratic_program, coefficients_list)

        values_list = list(range(len(x)))
        values_array = np.array(values_list)
        values_dict_int = {i: i for i in range(len(x))}
        values_dict_str = {'x{}'.format(i): i for i in range(len(x))}

        for values in [values_list, values_array, values_dict_int, values_dict_str]:
            np.testing.assert_almost_equal(linear.evaluate_gradient(values), coefficients_list)


if __name__ == '__main__':
    unittest.main()
