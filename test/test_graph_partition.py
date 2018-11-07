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

from test.common import QiskitAquaTestCase
from qiskit_aqua import run_algorithm, get_algorithm_instance, local_pluggables
from qiskit_aqua.input import get_input_instance
from qiskit_aqua.translators.ising import graphpartition


class TestGraphPartition(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):

        np.random.seed(100)
        self.w = graphpartition.random_graph(4, edge_prob=0.8, weight_range=10)
        self.qubit_op, self.offset = graphpartition.get_graphpartition_qubitops(self.w)
        self.algo_input = get_input_instance('EnergyInput')
        self.algo_input.qubit_op = self.qubit_op

    def test_graph_partition(self):
        algorithm_cfg = {
            'name': 'VQE',
            'operator_mode': 'matrix'
        }

        optimizer_cfg = {
            'name': 'SPSA',
            'max_trials': 300
        }

        var_form_cfg = {
            'name': 'RY',
            'depth': 5,
            'entanglement': 'linear'
        }

        params = {
            'problem': {'name': 'ising', 'random_seed': 10598},
            'algorithm': algorithm_cfg,
            'optimizer': optimizer_cfg,
            'variational_form': var_form_cfg,
            'backend': {'name': 'statevector_simulator'}
        }
        result = run_algorithm(params, self.algo_input)
        x = graphpartition.sample_most_likely(result['eigvecs'][0])

        # use the brute-force way to generate the oracle
        def bitfield(n, L):
            result = np.binary_repr(n, L)
            return [int(digit) for digit in result] # [2:] to chop off the "0b" part

        L = len(x)
        max = 2**len(x)
        minimal_conf = None
        minimal_v = np.inf
        for i in range(max):
            cur = bitfield(i, L)

            how_many_nonzero = np.count_nonzero(cur)
            if how_many_nonzero *2 != L: # not balanced
                continue

            cur_v = graphpartition.objective_value(np.array(cur), self.w)
            if cur_v < minimal_v:
                minimal_v = cur_v
                minimal_conf = cur

        # check against the oracle
        print("quantum solution", graphpartition.get_graph_solution(x))
        self.assertEqual(graphpartition.objective_value(x, self.w), minimal_v)



