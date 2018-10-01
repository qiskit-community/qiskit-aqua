# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import unittest

import numpy as np

from test.common import QiskitAquaTestCase
from qiskit_aqua import run_algorithm, get_algorithm_instance, local_pluggables
from qiskit_aqua.input import get_input_instance
from qiskit_aqua.translators.ising import maxcut


class TestCplexIsing(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        if 'CPLEX.Ising' not in local_pluggables('algorithm'):
            self.skipTest('CPLEX.Ising algorithm not found - CPLEX not installed?')

        np.random.seed(8123179)
        self.w = maxcut.random_graph(4, edge_prob=0.5, weight_range=10)
        self.qubit_op, self.offset = maxcut.get_maxcut_qubitops(self.w)
        self.algo_input = get_input_instance('EnergyInput')
        self.algo_input.qubit_op = self.qubit_op

    def test_cplex_ising_via_run_algorithm(self):
        params = {
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'CPLEX.Ising', 'display': 0}
        }
        result = run_algorithm(params, self.algo_input)
        self.assertEqual(result['energy'], -20.5)
        x_dict = result['x_sol']
        x = np.array([x_dict[i] for i in sorted(x_dict.keys())])
        np.testing.assert_array_equal(maxcut.get_graph_solution(x), [1, 0, 1, 1])
        self.assertEqual(maxcut.maxcut_value(x, self.w), 24)

    def test_cplex_ising_direct(self):
        algo = get_algorithm_instance('CPLEX.Ising')
        algo.init_args(self.algo_input.qubit_op, display=0)
        result = algo.run()
        self.assertEqual(result['energy'], -20.5)
        x_dict = result['x_sol']
        x = np.array([x_dict[i] for i in sorted(x_dict.keys())])
        np.testing.assert_array_equal(maxcut.get_graph_solution(x), [1, 0, 1, 1])
        self.assertEqual(maxcut.maxcut_value(x, self.w), 24)
