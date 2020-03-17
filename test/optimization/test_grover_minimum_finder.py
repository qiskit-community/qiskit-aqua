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
from docplex.mp.model import Model
from qiskit.finance.applications.ising import portfolio
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.optimization.algorithms import GroverMinimumFinder, MinimumEigenOptimizer
from qiskit.optimization.problems import OptimizationProblem


class TestGroverMinimumFinder(QiskitOptimizationTestCase):
    """GroverMinimumFinder Tests"""

    def validate_results(self, problem, results, max_iterations):
        """Validate the results object returned by GroverMinimumFinder."""
        # Get measured values.
        grover_results = results.results['grover_results']
        op_key = results.x
        op_value = results.fval
        iterations = len(grover_results.operation_counts)
        rot = grover_results.rotation_count
        func = grover_results.func_dict
        print("Function", func)
        print("Optimum Key:", op_key, "Optimum Value:", op_value, "Rotations:", rot, "\n")

        # Get expected value.
        solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        comp_result = solver.solve(problem)
        max_rotations = int(np.ceil(100*np.pi/4))

        # Validate results.
        max_hit = max_rotations <= rot or max_iterations <= iterations
        self.assertTrue(comp_result.x == results.x or max_hit)
        self.assertTrue(comp_result.fval == results.fval or max_hit)

    def test_qubo_gas_int_zero(self):
        """ Test for when the answer is zero """

        # Input.
        op = OptimizationProblem()
        op.variables.add(names=["x0", "x1"], types='BB')
        linear = [("x0", 0), ("x1", 0)]
        op.objective.set_linear(linear)

        # Will not find a negative, should return 0.
        gmf = GroverMinimumFinder(num_iterations=1)
        results = gmf.solve(op)
        self.assertEqual(results.x, [0, 0])
        self.assertEqual(results.fval, 0.0)

    def test_qubo_gas_int_simple(self):
        """ Test for simple case, with 2 linear coeffs and no quadratic coeffs or constants """

        # Input.
        op = OptimizationProblem()
        op.variables.add(names=["x0", "x1"], types='BB')
        linear = [("x0", -1), ("x1", 2)]
        op.objective.set_linear(linear)

        # Get the optimum key and value.
        n_iter = 8
        gmf = GroverMinimumFinder(num_iterations=n_iter)
        results = gmf.solve(op)
        self.validate_results(op, results, n_iter)

    def test_qubo_gas_int_paper_example(self):
        """ Test the example from https://arxiv.org/abs/1912.04088 """

        # Input.
        op = OptimizationProblem()
        op.variables.add(names=["x0", "x1", "x2"], types='BBB')

        linear = [("x0", -1), ("x1", 2), ("x2", -3)]
        op.objective.set_linear(linear)
        op.objective.set_quadratic_coefficients('x0', 'x2', -2)
        op.objective.set_quadratic_coefficients('x1', 'x2', -1)

        # Get the optimum key and value.
        n_iter = 10
        gmf = GroverMinimumFinder(num_iterations=n_iter)
        results = gmf.solve(op)
        self.validate_results(op, results, 10)
