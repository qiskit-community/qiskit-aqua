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

dj_in_0 = {'00': '0', '01': '0', '10': '0', '11': '0'}
dj_in_1 = {'00': '1', '01': '1', '10': '1', '11': '1'}
dj_in_2 = {'00': '0', '01': '1', '10': '0', '11': '1'}
dj_in_3 = {'000': '1', '001': '1', '010': '1', '011': '1',
           '100': '0', '101': '0', '110': '0', '111': '0'}

class TestDeutschJozsa(QiskitAquaTestCase):

    @parameterized.expand([
        [dj_in_0],
        [dj_in_1],
        [dj_in_2],
        [dj_in_3]
    ])

    def test_deutschjozsa(self, dj_input):
        dj_oracle = get_oracle_instance('DeutschJozsaOracle')
        dj = get_algorithm_instance('DeutschJozsa')
        dj.setup_quantum_backend(backend='qasm_simulator')
        dj.init_oracle(dj_oracle,dj_input)
        result = dj.run()
        self.assertTrue(result['oracle_evaluation'])

if __name__ == '__main__':
    unittest.main()