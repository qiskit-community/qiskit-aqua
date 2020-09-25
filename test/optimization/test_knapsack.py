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

""" Test Knapsack Problem """

import unittest
from test.optimization import QiskitOptimizationTestCase
import numpy as np

from qiskit.optimization.applications.ising import knapsack
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.aqua.algorithms import NumPyMinimumEigensolver


class TestTSP(QiskitOptimizationTestCase):
    """Knapsack Ising tests."""

    @staticmethod
    def _run_knapsack(values, weights, max_weight):
        qubit_op, _ = knapsack.get_operator(values, weights, max_weight)

        algo = NumPyMinimumEigensolver(qubit_op)
        result = algo.run()
        x = sample_most_likely(result.eigenstate)

        solution = knapsack.get_solution(x, values)
        value, weight = knapsack.knapsack_value_weight(solution, values, weights)

        return solution, value, weight

    def test_knapsack(self):
        """ Knapsack test """
        values = [10, 40, 50, 75]
        weights = [5, 10, 3, 12]
        max_weight = 20

        solution, value, weight = self._run_knapsack(values, weights, max_weight)

        np.testing.assert_array_equal(solution, [1, 0, 1, 1])
        np.testing.assert_equal(value, 135)
        np.testing.assert_equal(weight, 20)

    def test_knapsack_zero_max_weight(self):
        """ Knapsack zero max weight test """
        values = [10, 40, 50, 75]
        weights = [5, 10, 3, 12]
        max_weight = 0

        solution, value, weight = self._run_knapsack(values, weights, max_weight)

        np.testing.assert_array_equal(solution, [0, 0, 0, 0])
        np.testing.assert_equal(value, 0)
        np.testing.assert_equal(weight, 0)

    def test_knapsack_large_max_weight(self):
        """ Knapsack large max weight test """
        values = [10, 40, 50, 75]
        weights = [5, 10, 3, 12]
        max_weight = 1000

        solution, value, weight = self._run_knapsack(values, weights, max_weight)

        np.testing.assert_array_equal(solution, [1, 1, 1, 1])
        np.testing.assert_equal(value, 175)
        np.testing.assert_equal(weight, 30)


if __name__ == '__main__':
    unittest.main()
