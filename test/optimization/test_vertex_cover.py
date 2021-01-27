# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Vertex Cover """

import unittest
from test.optimization import QiskitOptimizationTestCase
import numpy as np
from qiskit import BasicAer
from qiskit.circuit.library import EfficientSU2

from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.optimization.applications.ising import vertex_cover
from qiskit.optimization.applications.ising.common import random_graph, sample_most_likely
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.aqua.components.optimizers import SPSA


class TestVertexCover(QiskitOptimizationTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        self.seed = 100
        aqua_globals.random_seed = self.seed
        self.num_nodes = 3
        self.w = random_graph(self.num_nodes, edge_prob=0.8, weight_range=10)
        self.qubit_op, self.offset = vertex_cover.get_operator(self.w)

    def _brute_force(self):
        # brute-force way
        def bitfield(n, length):
            result = np.binary_repr(n, length)
            return [int(digit) for digit in result]  # [2:] to chop off the "0b" part

        nodes = self.num_nodes
        maximum = 2 ** nodes
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
        """ Vertex Cover test """
        algo = NumPyMinimumEigensolver(self.qubit_op, aux_operators=[])
        result = algo.run()
        x = sample_most_likely(result.eigenstate)
        sol = vertex_cover.get_graph_solution(x)
        np.testing.assert_array_equal(sol, [0, 0, 1])
        oracle = self._brute_force()
        self.assertEqual(np.count_nonzero(sol), oracle)

    def test_vertex_cover_vqe(self):
        """ Vertex Cover VQE test """
        aqua_globals.random_seed = self.seed

        result = VQE(self.qubit_op,
                     EfficientSU2(reps=3),
                     SPSA(maxiter=200),
                     max_evals_grouped=2).run(
                         QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))

        x = sample_most_likely(result.eigenstate)
        sol = vertex_cover.get_graph_solution(x)
        oracle = self._brute_force()
        self.assertEqual(np.count_nonzero(sol), oracle)


if __name__ == '__main__':
    unittest.main()
