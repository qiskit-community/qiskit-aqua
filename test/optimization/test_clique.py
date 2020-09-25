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

""" Test Clique """

import unittest
from test.optimization import QiskitOptimizationTestCase
import numpy as np
from qiskit import BasicAer
from qiskit.circuit.library import RealAmplitudes

from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.optimization.applications.ising import clique
from qiskit.optimization.applications.ising.common import random_graph, sample_most_likely
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.aqua.components.optimizers import COBYLA


class TestClique(QiskitOptimizationTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        self.k = 5  # K means the size of the clique
        self.seed = 100
        aqua_globals.random_seed = self.seed
        self.num_nodes = 5
        self.w = random_graph(self.num_nodes, edge_prob=0.8, weight_range=10)
        self.qubit_op, self.offset = clique.get_operator(self.w, self.k)

    def _brute_force(self):
        # brute-force way: try every possible assignment!
        def bitfield(n, length):
            result = np.binary_repr(n, length)
            return [int(digit) for digit in result]

        nodes = self.num_nodes  # length of the bitstring that represents the assignment
        maximum = 2 ** nodes
        has_sol = False
        for i in range(maximum):
            cur = bitfield(i, nodes)
            cur_v = clique.satisfy_or_not(np.array(cur), self.w, self.k)
            if cur_v:
                has_sol = True
                break
        return has_sol

    def test_clique(self):
        """ Clique test """
        algo = NumPyMinimumEigensolver(self.qubit_op, aux_operators=[])
        result = algo.run()
        x = sample_most_likely(result.eigenstate)
        ising_sol = clique.get_graph_solution(x)
        np.testing.assert_array_equal(ising_sol, [1, 1, 1, 1, 1])
        oracle = self._brute_force()
        self.assertEqual(clique.satisfy_or_not(ising_sol, self.w, self.k), oracle)

    def test_clique_vqe(self):
        """ VQE Clique test """
        aqua_globals.random_seed = 10598
        result = VQE(self.qubit_op,
                     RealAmplitudes(reps=5, entanglement='linear'),
                     COBYLA(),
                     max_evals_grouped=2).run(
                         QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        x = sample_most_likely(result.eigenstate)
        ising_sol = clique.get_graph_solution(x)
        np.testing.assert_array_equal(ising_sol, [1, 1, 1, 1, 1])
        oracle = self._brute_force()
        self.assertEqual(clique.satisfy_or_not(ising_sol, self.w, self.k), oracle)


if __name__ == '__main__':
    unittest.main()
