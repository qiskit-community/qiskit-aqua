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
from qiskit_aqua.translators.ising import clique



class TestClique(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        self.K = 5 # K means the size of the clique
        np.random.seed(100)
        self.w = clique.random_graph(5, edge_prob=0.8, weight_range=10)
        from .drawutil import draw_graph_ndarray
        draw_graph_ndarray(self.w)
        self.qubit_op, self.offset = clique.get_clique_qubitops(self.w, self.K)
        self.algo_input = get_input_instance('EnergyInput')
        self.algo_input.qubit_op = self.qubit_op

    def test_clique(self):
        params = {
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, self.algo_input)
        x = clique.sample_most_likely(len(self.w), result['eigvecs'][0])

        # brute-force way: try every possible assignment!
        def bitfield(n, L):
            result = np.binary_repr(n, L)
            return [int(digit) for digit in result] # [2:] to chop off the "0b" part

        L = len(x)
        max = 2**len(x)
        has_sol = False
        for i in range(max):
            cur = bitfield(i, L)
            cur_v = clique.satisfy_or_not(np.array(cur), self.w, self.K)
            if cur_v:
                has_sol = True
                print("satisfiable assigment:", cur)
                break

        vqe_sol = clique.get_graph_solution(x)
        print('solution:', vqe_sol)
        print('solution satisfiability:', clique.satisfy_or_not(vqe_sol, self.w, self.K))

        # check against the oracle
        self.assertEqual(clique.satisfy_or_not(vqe_sol, self.w, self.K), has_sol)













