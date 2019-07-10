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

import numpy as np
import json

from test.common import QiskitAquaTestCase
from qiskit import BasicAer
from qiskit.aqua import run_algorithm
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.translators.ising import exact_cover
from qiskit.aqua.algorithms import ExactEigensolver


class TestExactCover(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        input_file = self._get_resource_path('sample.exactcover')
        with open(input_file) as f:
            self.list_of_subsets = json.load(f)
            qubitOp, offset = exact_cover.get_exact_cover_qubitops(self.list_of_subsets)
            self.algo_input = EnergyInput(qubitOp)

    def brute_force(self):
        # brute-force way: try every possible assignment!
        has_sol = False

        def bitfield(n, L):
            result = np.binary_repr(n, L)
            return [int(digit) for digit in result]  # [2:] to chop off the "0b" part

        L = len(self.list_of_subsets)
        max = 2**L
        for i in range(max):
            cur = bitfield(i, L)
            cur_v = exact_cover.check_solution_satisfiability(cur, self.list_of_subsets)
            if cur_v:
                has_sol = True
                break
        return has_sol

    def test_exact_cover(self):
        params = {
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, self.algo_input)
        x = exact_cover.sample_most_likely(len(self.list_of_subsets), result['eigvecs'][0])
        ising_sol = exact_cover.get_solution(x)
        np.testing.assert_array_equal(ising_sol, [0, 1, 1, 0])
        oracle = self.brute_force()
        self.assertEqual(exact_cover.check_solution_satisfiability(ising_sol, self.list_of_subsets), oracle)

    def test_exact_cover_direct(self):
        algo = ExactEigensolver(self.algo_input.qubit_op, k=1, aux_operators=[])
        result = algo.run()
        x = exact_cover.sample_most_likely(len(self.list_of_subsets), result['eigvecs'][0])
        ising_sol = exact_cover.get_solution(x)
        np.testing.assert_array_equal(ising_sol, [0, 1, 1, 0])
        oracle = self.brute_force()
        self.assertEqual(exact_cover.check_solution_satisfiability(ising_sol, self.list_of_subsets), oracle)

    def test_exact_cover_vqe(self):
        algorithm_cfg = {
            'name': 'VQE',
            'operator_mode': 'matrix',
            'max_evals_grouped': 2
        }

        optimizer_cfg = {
            'name': 'COBYLA'
        }

        var_form_cfg = {
            'name': 'RYRZ',
            'depth': 5
        }

        params = {
            'problem': {'name': 'ising', 'random_seed': 10598},
            'algorithm': algorithm_cfg,
            'optimizer': optimizer_cfg,
            'variational_form': var_form_cfg
        }
        backend = BasicAer.get_backend('statevector_simulator')
        result = run_algorithm(params, self.algo_input, backend=backend)
        x = exact_cover.sample_most_likely(len(self.list_of_subsets), result['eigvecs'][0])
        ising_sol = exact_cover.get_solution(x)
        oracle = self.brute_force()
        self.assertEqual(exact_cover.check_solution_satisfiability(ising_sol, self.list_of_subsets), oracle)
