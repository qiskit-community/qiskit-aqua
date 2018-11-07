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

import unittest

import numpy as np
import json

from test.common import QiskitAquaTestCase
from qiskit_aqua import run_algorithm, get_algorithm_instance, local_pluggables
from qiskit_aqua.input import get_input_instance
from qiskit_aqua.translators.ising import exactcover



class TestExactCover(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        pass


    def test_exactcover(self):
        with open('sample.exactcover') as f:
            list_of_subsets = json.load(f)
            qubitOp, offset = exactcover.get_exactcover_qubitops(list_of_subsets)
            algo_input = get_input_instance('EnergyInput')
            algo_input.qubit_op = qubitOp
            print(list_of_subsets)

        params = {
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, algo_input)


        # brute-force way: try every possible assignment!
        has_sol = False
        def bitfield(n, L):
            result = np.binary_repr(n, L)
            return [int(digit) for digit in result] # [2:] to chop off the "0b" part

        L = len(list_of_subsets)
        max = 2**L

        for i in range(max):
            cur = bitfield(i, L)

            cur_v = exactcover.check_solution_satisfiability(cur, list_of_subsets)
            if cur_v:
                has_sol = True
                print("satisfiable assigment:", cur)
                break

        x = exactcover.sample_most_likely(len(list_of_subsets), result['eigvecs'][0])
        sol = exactcover.get_solution(x)
        print('solution:', sol)
        print('solution satisfiability:', exactcover.check_solution_satisfiability(sol, list_of_subsets))


        self.assertEqual(exactcover.check_solution_satisfiability(sol, list_of_subsets), has_sol)
























