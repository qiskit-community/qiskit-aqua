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

""" Text Vertex Cover """

from test.aqua.common import QiskitAquaTestCase

import numpy as np
from qiskit import BasicAer

from qiskit.aqua import run_algorithm, aqua_globals
from qiskit.aqua.input import EnergyInput
from qiskit.optimization.ising import vertex_cover
from qiskit.optimization.ising.common import random_graph, sample_most_likely
from qiskit.aqua.algorithms import ExactEigensolver


class TestVertexCover(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        self.seed = 100
        aqua_globals.random_seed = self.seed
        self.num_nodes = 3
        self.w = random_graph(self.num_nodes, edge_prob=0.8, weight_range=10)
        self.qubit_op, self.offset = vertex_cover.get_operator(self.w)
        self.algo_input = EnergyInput(self.qubit_op)

    def _brute_force(self):
        # brute-force way
        def bitfield(n, length):
            result = np.binary_repr(n, length)
            return [int(digit) for digit in result]  # [2:] to chop off the "0b" part

        nodes = self.num_nodes
        maximum = 2**nodes
        minimal_v = np.inf
        for i in range(maximum):
            cur = bitfield(i, nodes)

            cur_v = vertex_cover.check_full_edge_coverage(np.array(cur), self.w)
            if cur_v:
                nonzerocount = np.count_nonzero(cur)
                if nonzerocount < minimal_v:
                    minimal_v = nonzerocount

        return minimal_v

    def test_vertex_cover(self):
        """ Vertex cover test """
        params = {
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, self.algo_input)

        x = sample_most_likely(result['eigvecs'][0])
        sol = vertex_cover.get_graph_solution(x)
        np.testing.assert_array_equal(sol, [0, 1, 1])
        oracle = self._brute_force()
        self.assertEqual(np.count_nonzero(sol), oracle)

    def test_vertex_cover_direct(self):
        """ Vertex Cover Direct test """
        algo = ExactEigensolver(self.algo_input.qubit_op, k=1, aux_operators=[])
        result = algo.run()
        x = sample_most_likely(result['eigvecs'][0])
        sol = vertex_cover.get_graph_solution(x)
        np.testing.assert_array_equal(sol, [0, 1, 1])
        oracle = self._brute_force()
        self.assertEqual(np.count_nonzero(sol), oracle)

    def test_vertex_cover_vqe(self):
        """ Vertex Cover VQE test """
        algorithm_cfg = {
            'name': 'VQE',
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
            'problem': {'name': 'ising', 'random_seed': self.seed},
            'algorithm': algorithm_cfg,
            'optimizer': optimizer_cfg,
            'variational_form': var_form_cfg
        }
        backend = BasicAer.get_backend('qasm_simulator')
        result = run_algorithm(params, self.algo_input, backend=backend)
        x = sample_most_likely(result['eigvecs'][0])
        sol = vertex_cover.get_graph_solution(x)
        oracle = self._brute_force()
        self.assertEqual(np.count_nonzero(sol), oracle)
