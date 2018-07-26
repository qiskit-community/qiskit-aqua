# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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
from parameterized import parameterized
from qiskit_aqua import get_algorithm_instance, get_optimizer_instance
from test.common import QiskitAquaTestCase
from qiskit_aqua.ising import maxcut
import numpy as np

w1 = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])
p1 = 1
s1 = {'0101', '1010'}


w2 = np.array([
    [0., 8., -9., 0.],
    [8., 0., 7., 9.],
    [-9., 7., 0., -8.],
    [0., 9., -8., 0.],
])
p2 = 1
s2 = {'1011', '0100'}


class TestQAOA(QiskitAquaTestCase):
    """Test QAOA with MaxCut."""
    @parameterized.expand([
        [w1, p1, s1],
        [w2, p2, s2],
    ])
    def test_qaoa(self, w, p, solutions):
        self.log.debug('Testing {}-step QAOA with MaxCut on graph\n{}'.format(p, w))
        np.random.seed(0)
        optimizer = get_optimizer_instance('COBYLA')
        qubitOp, offset = maxcut.get_maxcut_qubitops(w)
        qaoa = get_algorithm_instance('QAOA')
        qaoa.init_args(qubitOp, 'matrix', p, optimizer)
        qaoa.setup_quantum_backend(backend='local_statevector_simulator', shots=100)

        result = qaoa.run()
        x = maxcut.sample_most_likely(result['eigvecs'][0])
        graph_solution = maxcut.get_graph_solution(x)
        self.log.debug('energy:             {}'.format(result['energy']))
        self.log.debug('time:               {}'.format(result['eval_time']))
        self.log.debug('maxcut objective:   {}'.format(result['energy'] + offset))
        self.log.debug('solution:           {}'.format(graph_solution))
        self.log.debug('solution objective: {}'.format(maxcut.maxcut_value(x, w)))
        self.assertIn(''.join([str(int(i)) for i in graph_solution]), solutions)


if __name__ == '__main__':
    unittest.main()
