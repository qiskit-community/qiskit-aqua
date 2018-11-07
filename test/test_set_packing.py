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
from qiskit_aqua.translators.ising import setpacking



class TestSetPacking(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        pass


    def test_set_packing(self):
        with open('sample.setpacking') as f:
            list_of_subsets = json.load(f)
            qubitOp, offset = setpacking.get_setpacking_qubitops(list_of_subsets)
            algo_input = get_input_instance('EnergyInput')
            algo_input.qubit_op = qubitOp

        params = {
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, algo_input)
        x = setpacking.sample_most_likely(len(list_of_subsets), result['eigvecs'][0])
        sol = setpacking.get_solution(x)
        print('solution:', sol)
        print('check disjoint:', setpacking.check_disjoint(sol, list_of_subsets))


        # brute-force way: try every possible assignment!
        def bitfield(n, L):
            result = np.binary_repr(n, L)
            return [int(digit) for digit in result] # [2:] to chop off the "0b" part

        L = len(list_of_subsets)
        max = 2**L
        max_v = -np.inf
        max_assign = None
        for i in range(max):
            cur = bitfield(i, L)
            cur_v = setpacking.check_disjoint(cur, list_of_subsets)
            if cur_v:
                if np.count_nonzero(cur) > max_v:
                    max_v = np.count_nonzero(cur)
                    max_assign = cur


        self.assertEqual(np.count_nonzero(sol), max_v)

















