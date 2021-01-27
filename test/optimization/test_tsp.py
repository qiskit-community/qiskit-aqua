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

""" Test TSP (Traveling Salesman Problem) """

import unittest
from test.optimization import QiskitOptimizationTestCase
import numpy as np

from qiskit.aqua import aqua_globals
from qiskit.optimization.applications.ising import tsp
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.aqua.algorithms import NumPyMinimumEigensolver


class TestTSP(QiskitOptimizationTestCase):
    """TSP Ising tests."""

    def setUp(self):
        super().setUp()
        self.seed = 80598
        aqua_globals.random_seed = self.seed
        self.num_nodes = 3
        self.ins = tsp.random_tsp(self.num_nodes)
        self.qubit_op, self.offset = tsp.get_operator(self.ins)

    def test_tsp(self):
        """ TSP test """
        algo = NumPyMinimumEigensolver(self.qubit_op)
        result = algo.run()
        x = sample_most_likely(result.eigenstate)
        # print(self.qubit_op.to_opflow().eval(result.eigenstate).adjoint().eval(result.eigenstate))
        order = tsp.get_tsp_solution(x)
        np.testing.assert_equal(tsp.tsp_value(order, self.ins.w),
                                tsp.tsp_value([1, 2, 0], self.ins.w))

    def test_tsp_get_solution(self):
        """ Test tsp.get_tsp_solution()"""
        feasible = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        self.assertListEqual(tsp.get_tsp_solution(feasible), [0, 1, 2])
        infeasible = [1, 0, 0, 1, 1, 0, 0, 0, 0]
        self.assertListEqual(tsp.get_tsp_solution(infeasible), [[0, 1], 1, []])


if __name__ == '__main__':
    unittest.main()
