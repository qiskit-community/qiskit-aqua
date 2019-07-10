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
from qiskit.aqua import run_algorithm
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.translators.ising import set_packing
from qiskit.aqua.algorithms import ExactEigensolver


class TestSetPacking(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        input_file = self._get_resource_path('sample.setpacking')
        with open(input_file) as f:
            self.list_of_subsets = json.load(f)
            qubitOp, offset = set_packing.get_set_packing_qubitops(self.list_of_subsets)
            self.algo_input = EnergyInput(qubitOp)

    def brute_force(self):
        # brute-force way: try every possible assignment!
        def bitfield(n, L):
            result = np.binary_repr(n, L)
            return [int(digit) for digit in result]  # [2:] to chop off the "0b" part

        L = len(self.list_of_subsets)
        max = 2**L
        max_v = -np.inf
        for i in range(max):
            cur = bitfield(i, L)
            cur_v = set_packing.check_disjoint(cur, self.list_of_subsets)
            if cur_v:
                if np.count_nonzero(cur) > max_v:
                    max_v = np.count_nonzero(cur)
        return max_v

    def test_set_packing(self):
        params = {
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, self.algo_input)
        x = set_packing.sample_most_likely(len(self.list_of_subsets), result['eigvecs'][0])
        ising_sol = set_packing.get_solution(x)
        np.testing.assert_array_equal(ising_sol, [0, 1, 1])
        oracle = self.brute_force()
        self.assertEqual(np.count_nonzero(ising_sol), oracle)

    def test_set_packing_direct(self):
        algo = ExactEigensolver(self.algo_input.qubit_op, k=1, aux_operators=[])
        result = algo.run()
        x = set_packing.sample_most_likely(len(self.list_of_subsets), result['eigvecs'][0])
        ising_sol = set_packing.get_solution(x)
        np.testing.assert_array_equal(ising_sol, [0, 1, 1])
        oracle = self.brute_force()
        self.assertEqual(np.count_nonzero(ising_sol), oracle)

    def test_set_packing_vqe(self):
        try:
            from qiskit import Aer
        except Exception as e:
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(e)))
            return

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
            'name': 'RY',
            'depth': 5,
            'entanglement': 'linear'
        }

        params = {
            'problem': {'name': 'ising', 'random_seed': 100},
            'algorithm': algorithm_cfg,
            'optimizer': optimizer_cfg,
            'variational_form': var_form_cfg
        }
        backend = Aer.get_backend('qasm_simulator')
        result = run_algorithm(params, self.algo_input, backend=backend)
        x = set_packing.sample_most_likely(len(self.list_of_subsets), result['eigvecs'][0])
        ising_sol = set_packing.get_solution(x)
        oracle = self.brute_force()
        self.assertEqual(np.count_nonzero(ising_sol), oracle)
