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
from cplex import SparsePair, SparseTriple
from qiskit.optimization.algorithms import GroverMinimumFinder
from qiskit.optimization.problems import OptimizationProblem
from qiskit.optimization.util import get_qubo_solutions


class TestGroverMinimumFinder(QiskitOptimizationTestCase):
    """GroverMinimumFinder Tests"""

    def validate_results(self, results, max_iterations):
        """Validate the results object returned by GroverMinimumFinder."""
        # Get measured values.
        grover_results = results.results
        n_key = grover_results.n_input_qubits
        op_key = results.x
        op_value = results.fval
        iterations = len(grover_results.operation_counts)
        rot = grover_results.rotation_count
        func = grover_results.func_dict
        print("Function", func)
        print("Optimum Key:", op_key, "Optimum Value:", op_value, "Rotations:", rot, "\n")

        # Get expected value.
        solutions = get_qubo_solutions(func, n_key, print_solutions=False)
        min_key = min(solutions, key=lambda key: int(solutions[key]))
        min_value = solutions[min_key]
        max_rotations = int(np.ceil(100*np.pi/4))

        # Validate results.
        max_hit = max_rotations <= rot or max_iterations <= iterations
        self.assertTrue(min_key == op_key or max_hit)
        self.assertTrue(min_value == op_value or max_hit)

    def test_qubo_gas_int_zero(self):
        """ Test for when the answer is zero """
        # Circuit parameters.
        num_value = 4

        # Input.
        op = OptimizationProblem()
        _ = op.variables.add(names=["x0", "x1"])
        x0_linear = SparsePair(ind=['x0'], val=[0])
        x1_linear = SparsePair(ind=['x1'], val=[0])
        op.linear_constraints.add(lin_expr=[x0_linear, x1_linear])

        # Will not find a negative, should return 0.
        gmf = GroverMinimumFinder(num_iterations=1)
        results = gmf.solve(op)
        self.assertEqual(results.x, 0)
        self.assertEqual(int(results.fval), 0)

    def test_qubo_gas_int_simple(self):
        """ Test for simple case, with 2 linear coeffs and no quadratic coeffs or constants """
        # Circuit parameters.
        num_value = 4

        # Input.
        op = OptimizationProblem()
        _ = op.variables.add(names=["x0", "x1"])
        x0_linear = SparsePair(ind=['x0'], val=[-1])
        x1_linear = SparsePair(ind=['x1'], val=[2])
        op.linear_constraints.add(lin_expr=[x0_linear, x1_linear])

        # Get the optimum key and value.
        n_iter = 8
        gmf = GroverMinimumFinder(num_iterations=n_iter)
        results = gmf.solve(op)
        self.validate_results(results, n_iter)

    def test_qubo_gas_int_paper_example(self):
        """ Test the example from https://arxiv.org/abs/1912.04088 """
        # Circuit parameters.
        num_value = 5

        # Input.
        op = OptimizationProblem()
        _ = op.variables.add(names=["x0", "x1", "x2"])
        x0_linear = SparsePair(ind=['x0'], val=[-1])
        x1_linear = SparsePair(ind=['x1'], val=[2])
        x2_linear = SparsePair(ind=['x2'], val=[-3])
        x0_x2 = SparseTriple(ind1=['x0'], ind2=['x2'], val=[-2])
        x1_x2 = SparseTriple(ind1=['x1'], ind2=['x2'], val=[-1])
        op.quadratic_constraints.add(name='x0x2', quad_expr=x0_x2)
        op.quadratic_constraints.add(name='x1x2', quad_expr=x1_x2)
        op.linear_constraints.add(lin_expr=[x0_linear, x1_linear, x2_linear])

        # Get the optimum key and value.
        n_iter = 10
        gmf = GroverMinimumFinder(num_iterations=n_iter)
        results = gmf.solve(op)
        self.validate_results(results, 10)
