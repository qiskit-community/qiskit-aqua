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
from parameterized import parameterized
from test.common import QiskitAquaTestCase

from qiskit_aqua import get_algorithm_instance, get_oracle_instance

simon_000 = {
    '000': '001',
    '001': '010',
    '010': '011',
    '011': '100',

    '100': '101',
    '101': '110',
    '110': '111',
    '111': '000'
}

simon_110 = {
    '000': '101',
    '001': '010',
    '010': '000',
    '011': '110',

    '100': '000',
    '101': '110',
    '110': '101',
    '111': '010'
}

class TestSimon(QiskitAquaTestCase):

    @parameterized.expand([
        [simon_000],
        [simon_110]
    ])

    def test_simon(self, bv_input):
        simon_oracle = get_oracle_instance('SimonOracle')
        simon = get_algorithm_instance('Simon')
        simon.setup_quantum_backend(backend='qasm_simulator')
        simon.init_oracle(simon_oracle,simon_input)
        result = simon.run()
        self.assertTrue(result['oracle_evaluation'])

if __name__ == '__main__':
    unittest.main()