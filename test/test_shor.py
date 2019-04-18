# -*- coding: utf-8 -*-

# Copyright 2019 IBM RESEARCH. All Rights Reserved.
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

import unittest, math

from parameterized import parameterized
from qiskit import BasicAer

from qiskit.aqua import run_algorithm, QuantumInstance, AquaError
from qiskit.aqua.algorithms import Shor
from test.common import QiskitAquaTestCase


class TestShor(QiskitAquaTestCase):
    """test Shor's algorithm"""

    @parameterized.expand([
        [15, 'qasm_simulator', [3, 5]],
    ])
    def test_shor_factoring(self, N, backend, factors):
        params = {
            'problem': {
                'name': 'factoring',
            },
            'algorithm': {
                'name': 'Shor',
                'N': N,
            },
            'backend': {
                'shots': 1000,
            },
        }
        result_dict = run_algorithm(params, backend=BasicAer.get_backend(backend))
        self.assertListEqual(result_dict['factors'][0], factors)

    @parameterized.expand([
        [5],
        [7],
    ])
    def test_shor_no_factors(self, N):
        shor = Shor(N)
        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1000)
        ret = shor.run(quantum_instance)
        self.assertTrue(ret['factors'] == [])

    @parameterized.expand([
        [3, 5],
        [5, 3],
    ])
    def test_shor_power(self, base, power):
        N = int(math.pow(base, power))
        shor = Shor(N)
        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1000)
        ret = shor.run(quantum_instance)
        self.assertTrue(ret['factors'] == [base])

    @parameterized.expand([
        [-1],
        [0],
        [1],
        [2],
        [4],
        [16],
    ])
    def test_shor_bad_input(self, N):
        with self.assertRaises(AquaError):
            Shor(N)


if __name__ == '__main__':
    unittest.main()
