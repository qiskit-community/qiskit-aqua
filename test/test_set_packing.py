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
import json

from test.common import QiskitAquaTestCase
from qiskit import BasicAer
from qiskit.aqua import run_algorithm
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.translators.ising import setpacking
from qiskit.aqua.algorithms import ExactEigensolver


class TestSetPacking(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        input_file = self._get_resource_path('sample.setpacking')
        with open(input_file) as f:
            self.list_of_subsets = json.load(f)
            qubitOp, offset = setpacking.get_setpacking_qubitops(self.list_of_subsets)
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
            cur_v = setpacking.check_disjoint(cur, self.list_of_subsets)
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
        x = setpacking.sample_most_likely(len(self.list_of_subsets), result['eigvecs'][0])
        ising_sol = setpacking.get_solution(x)
        np.testing.assert_array_equal(ising_sol, [0, 1, 1])
        oracle = self.brute_force()
        self.assertEqual(np.count_nonzero(ising_sol), oracle)

    def test_set_packing_direct(self):
        algo = ExactEigensolver(self.algo_input.qubit_op, k=1, aux_operators=[])
        result = algo.run()
        x = setpacking.sample_most_likely(len(self.list_of_subsets), result['eigvecs'][0])
        ising_sol = setpacking.get_solution(x)
        np.testing.assert_array_equal(ising_sol, [0, 1, 1])
        oracle = self.brute_force()
        self.assertEqual(np.count_nonzero(ising_sol), oracle)

    # TODO Disable for now until Aer is ok
    def todo_test_set_packing_vqe(self):
        algorithm_cfg = {
            'name': 'VQE',
            'operator_mode': 'grouped_paulis',
            'batch_mode': True
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
        backend = BasicAer.get_backend('qasm_simulator')
        result = run_algorithm(params, self.algo_input, backend=backend)
        x = setpacking.sample_most_likely(len(self.list_of_subsets), result['eigvecs'][0])
        ising_sol = setpacking.get_solution(x)
        oracle = self.brute_force()
        self.assertEqual(np.count_nonzero(ising_sol), oracle)
