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

from qiskit.aqua.components.oracles import DeutschJozsaOracle
from qiskit.aqua.algorithms import DeutschJozsa
from qiskit.aqua import get_aer_backend

from test.common import QiskitAquaTestCase


class TestDeutschJozsa(QiskitAquaTestCase):
    @parameterized.expand([
        [{'00': '0', '01': '0', '10': '0', '11': '0'}],
        [{'00': '1', '01': '1', '10': '1', '11': '1'}],
        [{'00': '0', '01': '1', '10': '0', '11': '1'}],
        [{'000': '1', '001': '1', '010': '1', '011': '1',
          '100': '0', '101': '0', '110': '0', '111': '0'}]
    ])
    def test_deutschjozsa(self, dj_input):
        backend = get_aer_backend('qasm_simulator')
        oracle = DeutschJozsaOracle(dj_input)
        algorithm = DeutschJozsa(oracle)
        result = algorithm.run(backend)
        if sum([int(v) for v in dj_input.values()]) == len(dj_input) / 2:
            self.assertTrue(result['result'] == 'balanced')
        else:
            self.assertTrue(result['result'] == 'constant')


if __name__ == '__main__':
    unittest.main()
