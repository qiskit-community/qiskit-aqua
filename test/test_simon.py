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

from qiskit.aqua.components.oracles import SimonOracle
from qiskit.aqua.algorithms import Simon
from qiskit.aqua import get_aer_backend

from test.common import QiskitAquaTestCase


class TestSimon(QiskitAquaTestCase):
    @parameterized.expand([
        [{'000': '001', '001': '010', '010': '011', '011': '100',
          '100': '101', '101': '110', '110': '111', '111': '000'}],
        [{'000': '101', '001': '010', '010': '000', '011': '110',
          '100': '000', '101': '110', '110': '101', '111': '010'}]
    ])
    def test_simon(self, simon_input):
        backend = get_aer_backend('qasm_simulator')
        oracle = SimonOracle(simon_input)
        algorithm = Simon(oracle)
        result = algorithm.run(backend)
        self.assertTrue(result['oracle_evaluation'])


if __name__ == '__main__':
    unittest.main()
