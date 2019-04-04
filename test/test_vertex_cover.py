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

import numpy as np

from test.common import QiskitAquaTestCase
from qiskit import BasicAer

from qiskit.aqua import run_algorithm
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.translators.ising import vertexcover
from qiskit.aqua.algorithms import ExactEigensolver


class TestVertexCover(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        np.random.seed(100)
        self.num_nodes = 3
        self.w = vertexcover.random_graph(self.num_nodes, edge_prob=0.8, weight_range=10)
        self.qubit_op, self.offset = vertexcover.get_vertexcover_qubitops(self.w)
        self.algo_input = EnergyInput(self.qubit_op)

    def brute_force(self):
        # brute-force way
        def bitfield(n, L):
            result = np.binary_repr(n, L)
            return [int(digit) for digit in result]  # [2:] to chop off the "0b" part

        L = self.num_nodes
        max = 2**L
        minimal_v = np.inf
        for i in range(max):
            cur = bitfield(i, L)

            cur_v = vertexcover.check_full_edge_coverage(np.array(cur), self.w)
            if cur_v:
                nonzerocount = np.count_nonzero(cur)
                if nonzerocount < minimal_v:
                    minimal_v = nonzerocount

        return minimal_v

    def test_vertex_cover(self):
        params = {
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, self.algo_input)

        x = vertexcover.sample_most_likely(len(self.w), result['eigvecs'][0])
        sol = vertexcover.get_graph_solution(x)
        np.testing.assert_array_equal(sol, [0, 1, 1])
        oracle = self.brute_force()
        self.assertEqual(np.count_nonzero(sol), oracle)

    def test_vertex_cover_direct(self):
        algo = ExactEigensolver(self.algo_input.qubit_op, k=1, aux_operators=[])
        result = algo.run()
        x = vertexcover.sample_most_likely(len(self.w), result['eigvecs'][0])
        sol = vertexcover.get_graph_solution(x)
        np.testing.assert_array_equal(sol, [0, 1, 1])
        oracle = self.brute_force()
        self.assertEqual(np.count_nonzero(sol), oracle)

    def test_vertex_cover_vqe(self):
        algorithm_cfg = {
            'name': 'VQE',
            'operator_mode': 'grouped_paulis',
            'max_evals_grouped': 2
        }

        optimizer_cfg = {
            'name': 'SPSA',
            'max_trials': 200
        }

        var_form_cfg = {
            'name': 'RYRZ',
            'depth': 3,
        }

        params = {
            'problem': {'name': 'ising', 'random_seed': 100},
            'algorithm': algorithm_cfg,
            'optimizer': optimizer_cfg,
            'variational_form': var_form_cfg
        }
        backend = BasicAer.get_backend('qasm_simulator')
        result = run_algorithm(params, self.algo_input, backend=backend)
        x = vertexcover.sample_most_likely(len(self.w), result['eigvecs'][0])
        sol = vertexcover.get_graph_solution(x)
        oracle = self.brute_force()
        self.assertEqual(np.count_nonzero(sol), oracle)
