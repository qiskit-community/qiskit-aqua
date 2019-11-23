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

""" Test Cplex Ising """

from test.optimization.common import QiskitOptimizationTestCase
import warnings
import numpy as np

from qiskit.aqua import run_algorithm, AquaError, aqua_globals
from qiskit.aqua.input import EnergyInput
from qiskit.optimization.ising import max_cut
from qiskit.optimization.ising.common import random_graph
from qiskit.aqua.algorithms.classical.cplex.cplex_ising import CPLEX_Ising


class TestCplexIsing(QiskitOptimizationTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        warnings.filterwarnings("ignore", message=aqua_globals.CONFIG_DEPRECATION_MSG,
                                category=DeprecationWarning)
        aqua_globals.random_seed = 8123179
        self.w = random_graph(4, edge_prob=0.5, weight_range=10)
        self.qubit_op, self.offset = max_cut.get_operator(self.w)
        self.algo_input = EnergyInput(self.qubit_op)

    def test_cplex_ising_via_run_algorithm(self):
        """ CPlex ising via run algorithm test """
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
        except AquaError as ex:
            self.skipTest(str(ex))

    def test_cplex_ising_direct(self):
        """ cplex ising direct test """
        try:
            algo = CPLEX_Ising(self.algo_input.qubit_op, display=0)
            result = algo.run()
            self.assertEqual(result['energy'], -20.5)
            x_dict = result['x_sol']
            x = np.array([x_dict[i] for i in sorted(x_dict.keys())])
            np.testing.assert_array_equal(
                max_cut.get_graph_solution(x), [1, 0, 1, 1])
            self.assertEqual(max_cut.max_cut_value(x, self.w), 24)
        except AquaError as ex:
            self.skipTest(str(ex))
