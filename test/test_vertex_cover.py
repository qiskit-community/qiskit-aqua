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
from qiskit_aqua.translators.ising import vertexcover



class TestVertexCover(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        np.random.seed(100)
        self.w = vertexcover.random_graph(4, edge_prob=0.8, weight_range=10)
        from .drawutil import draw_graph_ndarray
        draw_graph_ndarray(self.w)
        self.qubit_op, self.offset = vertexcover.get_vertexcover_qubitops(self.w)
        self.algo_input = get_input_instance('EnergyInput')
        self.algo_input.qubit_op = self.qubit_op



    def test_vertexcover(self):

        params = {
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, self.algo_input)


        # print('objective function:', maxcut.maxcut_obj(result, offset))
        x = vertexcover.sample_most_likely(len(self.w), result['eigvecs'][0])
        sol = vertexcover.get_graph_solution(x)
        print('solution:', sol)
        print('check full edge coverage:', vertexcover.check_full_edge_coverage(sol, self.w))

        # brute-force way
        def bitfield(n, L):
            result = np.binary_repr(n, L)
            return [int(digit) for digit in result] # [2:] to chop off the "0b" part

        L = len(x)
        max = 2**len(x)
        minimal_conf = None
        minimal_v = np.inf
        for i in range(max):
            cur = bitfield(i, L)

            cur_v = vertexcover.check_full_edge_coverage(np.array(cur), self.w)
            if cur_v:
                nonzerocount = np.count_nonzero(cur)
                if nonzerocount < minimal_v:
                    minimal_v = nonzerocount
                    minimal_conf = cur


        # print("minimal assigment:", minimal_conf)
        # print("minimal cardinality vertex cover", minimal_v)
        self.assertEqual(np.count_nonzero(sol), minimal_v)










