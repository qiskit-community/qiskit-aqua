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

""" Test Exact Cover """

import unittest
import json
from test.optimization import QiskitOptimizationTestCase
import numpy as np
from qiskit import BasicAer
from qiskit.circuit.library import EfficientSU2
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.optimization.applications.ising import exact_cover
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.aqua.components.optimizers import COBYLA


class TestExactCover(QiskitOptimizationTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        input_file = self.get_resource_path('sample.exactcover')
        with open(input_file) as file:
            self.list_of_subsets = json.load(file)
            self.qubit_op, _ = exact_cover.get_operator(self.list_of_subsets)

    def _brute_force(self):
        # brute-force way: try every possible assignment!
        has_sol = False

        def bitfield(n, length):
            result = np.binary_repr(n, length)
            return [int(digit) for digit in result]  # [2:] to chop off the "0b" part

        subsets = len(self.list_of_subsets)
        maximum = 2 ** subsets
        for i in range(maximum):
            cur = bitfield(i, subsets)
            cur_v = exact_cover.check_solution_satisfiability(cur, self.list_of_subsets)
            if cur_v:
                has_sol = True
                break
        return has_sol

    def test_exact_cover(self):
        """ Exact Cover test """
        algo = NumPyMinimumEigensolver(self.qubit_op, aux_operators=[])
        result = algo.run()
        x = sample_most_likely(result.eigenstate)
        ising_sol = exact_cover.get_solution(x)
        np.testing.assert_array_equal(ising_sol, [0, 1, 1, 0])
        oracle = self._brute_force()
        self.assertEqual(exact_cover.check_solution_satisfiability(ising_sol, self.list_of_subsets),
                         oracle)

    def test_exact_cover_vqe(self):
        """ Exact Cover VQE test """
        aqua_globals.random_seed = 10598
        result = VQE(self.qubit_op,
                     EfficientSU2(self.qubit_op.num_qubits, reps=5),
                     COBYLA(),
                     max_evals_grouped=2).run(
                         QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        x = sample_most_likely(result.eigenstate)
        ising_sol = exact_cover.get_solution(x)
        oracle = self._brute_force()
        self.assertEqual(exact_cover.check_solution_satisfiability(ising_sol, self.list_of_subsets),
                         oracle)


if __name__ == '__main__':
    unittest.main()
