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

""" Test Graph Partition """

from test.aqua.common import QiskitAquaTestCase
import numpy as np
from qiskit import BasicAer
from qiskit.aqua import run_algorithm, aqua_globals
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.translators.ising import graph_partition
from qiskit.aqua.translators.ising.common import random_graph, sample_most_likely
from qiskit.aqua.algorithms import ExactEigensolver


class TestGraphPartition(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        aqua_globals.random_seed = 100
        self.num_nodes = 4
        self.w = random_graph(self.num_nodes, edge_prob=0.8, weight_range=10)
        self.qubit_op, self.offset = graph_partition.get_qubit_op(self.w)
        self.algo_input = EnergyInput(self.qubit_op)

    def _brute_force(self):
        # use the brute-force way to generate the oracle
        def bitfield(n, length):
            result = np.binary_repr(n, length)
            return [int(digit) for digit in result]  # [2:] to chop off the "0b" part

        nodes = self.num_nodes
        maximum = 2**nodes
        minimal_v = np.inf
        for i in range(maximum):
            cur = bitfield(i, nodes)

            how_many_nonzero = np.count_nonzero(cur)
            if how_many_nonzero * 2 != nodes:  # not balanced
                continue

            cur_v = graph_partition.objective_value(np.array(cur), self.w)
            if cur_v < minimal_v:
                minimal_v = cur_v
        return minimal_v

    def test_graph_partition(self):
        """ Graph Partition test """
        params = {
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, self.algo_input)
        x = sample_most_likely(result['eigvecs'][0])
        # check against the oracle
        ising_sol = graph_partition.get_graph_solution(x)
        np.testing.assert_array_equal(ising_sol, [0, 1, 0, 1])
        oracle = self._brute_force()
        self.assertEqual(graph_partition.objective_value(x, self.w), oracle)

    def test_graph_partition_direct(self):
        """ Graph Partition Direct test """
        algo = ExactEigensolver(self.algo_input.qubit_op, k=1, aux_operators=[])
        result = algo.run()
        x = sample_most_likely(result['eigvecs'][0])
        # check against the oracle
        ising_sol = graph_partition.get_graph_solution(x)
        np.testing.assert_array_equal(ising_sol, [0, 1, 0, 1])
        oracle = self._brute_force()
        self.assertEqual(graph_partition.objective_value(x, self.w), oracle)

    def test_graph_partition_vqe(self):
        """ Graph Partition VQE test """
        algorithm_cfg = {
            'name': 'VQE',
            'max_evals_grouped': 2
        }

        optimizer_cfg = {
            'name': 'SPSA',
            'max_trials': 300
        }

        var_form_cfg = {
            'name': 'RY',
            'depth': 5,
            'entanglement': 'linear'
        }

        params = {
            'problem': {'name': 'ising', 'random_seed': 10598},
            'algorithm': algorithm_cfg,
            'optimizer': optimizer_cfg,
            'variational_form': var_form_cfg
        }
        backend = BasicAer.get_backend('statevector_simulator')
        result = run_algorithm(params, self.algo_input, backend=backend)
        x = sample_most_likely(result['eigvecs'][0])
        # check against the oracle
        ising_sol = graph_partition.get_graph_solution(x)
        np.testing.assert_array_equal(ising_sol, [0, 1, 0, 1])
        oracle = self._brute_force()
        self.assertEqual(graph_partition.objective_value(x, self.w), oracle)
