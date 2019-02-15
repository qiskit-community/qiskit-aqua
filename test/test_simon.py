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

import math
import numpy as np
from parameterized import parameterized
import unittest

from qiskit.aqua.components.oracles import TruthTableOracle
from qiskit.aqua.algorithms import Simon
from qiskit.aqua import get_aer_backend, AquaError

from test.common import QiskitAquaTestCase


class TestSimon(QiskitAquaTestCase):
    @parameterized.expand([
        [
            [
                '00011110',
                '01100110',
                '10101010',
            ]
        ],
        [
            [
                '10010110',
                '01010101',
                '10000010',
            ]
        ],
        [
            [
                '01101001',
                '10011001',
                '01100110',
            ]
        ],
    ])
    def test_simon(self, simon_input):
        # find the two keys that have matching values
        nbits = int(math.log(len(simon_input[0]), 2))
        vals = list(zip(*simon_input))[::-1]

        def find_pair():
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    if vals[i] == vals[j]:
                        return i, j
            return 0, 0

        k1, k2 = find_pair()
        hidden = np.binary_repr(k1 ^ k2, nbits)

        for optimization_mode in [None, 'simple']:
            backend = get_aer_backend('qasm_simulator')
            oracle = TruthTableOracle(simon_input, optimization_mode=optimization_mode)
            algorithm = Simon(oracle)
            result = algorithm.run(backend)
            # print(result['circuit'].draw(line_length=10000))
            self.assertEqual(result['result'], hidden)


if __name__ == '__main__':
    unittest.main()
