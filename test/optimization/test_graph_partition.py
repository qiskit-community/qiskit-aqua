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

""" Test Graph Partition """

import unittest
from test.optimization import QiskitOptimizationTestCase
import numpy as np
from qiskit import BasicAer
from qiskit.circuit.library import RealAmplitudes
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.optimization.applications.ising import graph_partition
from qiskit.optimization.applications.ising.common import random_graph, sample_most_likely
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.aqua.components.optimizers import SPSA


class TestGraphPartition(QiskitOptimizationTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        aqua_globals.random_seed = 100
        self.num_nodes = 4
        self.w = random_graph(self.num_nodes, edge_prob=0.8, weight_range=10)
        self.qubit_op, self.offset = graph_partition.get_operator(self.w)

    def _brute_force(self):
        # use the brute-force way to generate the oracle
        def bitfield(n, length):
            result = np.binary_repr(n, length)
            return [int(digit) for digit in result]  # [2:] to chop off the "0b" part

        nodes = self.num_nodes
        maximum = 2 ** nodes
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
        algo = NumPyMinimumEigensolver(self.qubit_op, aux_operators=[])
        result = algo.run()
        x = sample_most_likely(result.eigenstate)
        # check against the oracle
        ising_sol = graph_partition.get_graph_solution(x)
        # solutions are equivalent
        self.assertEqual(graph_partition.objective_value(np.array([0, 1, 0, 1]), self.w),
                         graph_partition.objective_value(ising_sol, self.w))
        oracle = self._brute_force()
        self.assertEqual(graph_partition.objective_value(x, self.w), oracle)

    def test_graph_partition_vqe(self):
        """ Graph Partition VQE test """
        aqua_globals.random_seed = 10213
        wavefunction = RealAmplitudes(self.qubit_op.num_qubits, insert_barriers=True,
                                      reps=5, entanglement='linear')
        result = VQE(self.qubit_op,
                     wavefunction,
                     SPSA(maxiter=300),
                     max_evals_grouped=2).run(
                         QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))

        x = sample_most_likely(result.eigenstate)
        # check against the oracle
        ising_sol = graph_partition.get_graph_solution(x)
        self.assertEqual(graph_partition.objective_value(np.array([0, 1, 0, 1]), self.w),
                         graph_partition.objective_value(ising_sol, self.w))
        oracle = self._brute_force()
        self.assertEqual(graph_partition.objective_value(x, self.w), oracle)


if __name__ == '__main__':
    unittest.main()
