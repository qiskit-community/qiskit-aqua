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

""" Test Grover Minimum Finder """

from test.optimization import QiskitOptimizationTestCase
import numpy as np
from qiskit.optimization.algorithms import GroverMinimumFinder
from qiskit.optimization.util import get_qubo_solutions

# Flag for verbosity in all units under test.
VERBOSE = False


class TestGroverMinimumFinder(QiskitOptimizationTestCase):
    """GroverMinimumFinder Tests"""

    def validate_results(self, results):
        """Validate the results object returned by GroverMinimumFinder."""
        # Get measured values.
        n_key = results.n_input_qubits
        op_key = results.optimum_input
        op_value = results.optimum_output
        rot = results.rotation_count
        func = results.func_dict
        print("Optimum Key:", op_key, "Optimum Value:", op_value, "Rotations:", rot, "\n")

        # Get expected value.
        solutions = get_qubo_solutions(func, n_key, print_solutions=False)
        min_key = min(solutions, key=lambda key: int(solutions[key]))
        min_value = solutions[min_key]
        max_rotations = np.ceil(2 ** (n_key / 2))

        # Validate results.
        self.assertTrue(min_key == op_key or max_rotations == rot)
        self.assertTrue(min_value == op_value or max_rotations == rot)

    def test_qubo_gas_int_zero(self):
        """ Test for when the answer is zero """
        # Circuit parameters.
        num_value = 4

        # Input.
        linear = np.array([0, 0])
        quadratic = np.array([[0, 0],
                              [0, 0]])
        constant = 0

        # Will not find a negative, should return 0.
        gmf = GroverMinimumFinder(num_iterations=1, verbose=VERBOSE)
        results = gmf.solve(quadratic, linear, constant, num_value)
        self.assertEqual(results.optimum_input, 0)
        self.assertEqual(int(results.optimum_output), 0)

    def test_qubo_gas_int_simple(self):
        """ Test for simple case, with 2 linear coeffs and no quadratic coeffs or constants """
        # Circuit parameters.
        num_value = 4

        # Input.
        linear = np.array([1, -2])
        quadratic = np.array([[2, 0],
                              [0, 2]])
        q = 0.5
        quadratic = quadratic.dot(q)

        # Get the optimum key and value.
        gmf = GroverMinimumFinder(num_iterations=6, verbose=VERBOSE)
        results = gmf.solve(quadratic, linear, 0, num_value)
        self.validate_results(results)

    def test_qubo_gas_int_simple_pos_constant(self):
        """ Test for a positive constant """
        # Circuit parameters.
        num_value = 4

        # Input.
        linear = np.array([1, -2])
        quadratic = np.array([[2, 0],
                              [0, 2]])
        q = 0.5
        quadratic = quadratic.dot(q)
        constant = 2

        # Get the optimum key and value.
        gmf = GroverMinimumFinder(num_iterations=6, verbose=VERBOSE)
        results = gmf.solve(quadratic, linear, constant, num_value)
        self.validate_results(results)

    def test_qubo_gas_int_simple_neg_constant(self):
        """ Test for a negative constant """
        # Circuit parameters.
        num_value = 4

        # Input.
        linear = np.array([1, -2])
        quadratic = np.array([[2, 0],
                              [0, 2]])
        q = 0.5
        quadratic = quadratic.dot(q)
        constant = -2

        # Get the optimum key and value.
        gmf = GroverMinimumFinder(num_iterations=6, verbose=VERBOSE)
        results = gmf.solve(quadratic, linear, constant, num_value)
        self.validate_results(results)

    def test_qubo_gas_int_paper_example(self):
        """ Test the example from https://arxiv.org/abs/1912.04088 """
        # Circuit parameters.
        num_value = 5

        # Input.
        linear = np.array([1, -2, 3])
        quadratic = np.array([[2, 0, -4],
                              [0, 4, -2],
                              [-4, -2, 10]])
        q = 0.5
        quadratic = quadratic.dot(q)

        # Get the optimum key and value.
        gmf = GroverMinimumFinder(num_iterations=20, verbose=VERBOSE)
        results = gmf.solve(quadratic, linear, 0, num_value)
        self.validate_results(results)
