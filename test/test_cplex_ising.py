# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np

from test.common import QiskitAquaTestCase
from qiskit.aqua import run_algorithm, AquaError
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.translators.ising import max_cut
from qiskit.aqua.algorithms.classical.cplex.cplex_ising import CPLEX_Ising


class TestCplexIsing(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        np.random.seed(8123179)
        self.w = max_cut.random_graph(4, edge_prob=0.5, weight_range=10)
        self.qubit_op, self.offset = max_cut.get_max_cut_qubitops(self.w)
        self.algo_input = EnergyInput(self.qubit_op)

    def test_cplex_ising_via_run_algorithm(self):
        try:
            params = {
                'problem': {'name': 'ising'},
                'algorithm': {'name': 'CPLEX.Ising', 'display': 0}
            }
            result = run_algorithm(params, self.algo_input)
            self.assertEqual(result['energy'], -20.5)
            x_dict = result['x_sol']
            x = np.array([x_dict[i] for i in sorted(x_dict.keys())])
            np.testing.assert_array_equal(
                max_cut.get_graph_solution(x), [1, 0, 1, 1])
            self.assertEqual(max_cut.max_cut_value(x, self.w), 24)
        except AquaError as e:
            self.skipTest(str(e))

    def test_cplex_ising_direct(self):
        try:
            algo = CPLEX_Ising(self.algo_input.qubit_op, display=0)
            result = algo.run()
            self.assertEqual(result['energy'], -20.5)
            x_dict = result['x_sol']
            x = np.array([x_dict[i] for i in sorted(x_dict.keys())])
            np.testing.assert_array_equal(
                max_cut.get_graph_solution(x), [1, 0, 1, 1])
            self.assertEqual(max_cut.max_cut_value(x, self.w), 24)
        except AquaError as e:
            self.skipTest(str(e))
