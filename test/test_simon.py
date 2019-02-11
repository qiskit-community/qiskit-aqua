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

from qiskit.aqua.components.oracles import ESOPOracle
from qiskit.aqua.algorithms import Simon
from qiskit.aqua import get_aer_backend, AquaError

from test.common import QiskitAquaTestCase


class TestSimon(QiskitAquaTestCase):
    @parameterized.expand([
        [{'000': '001', '001': '010', '010': '011', '011': '100',
          '100': '101', '101': '110', '110': '111', '111': '000'}],
        [{'000': '101', '001': '010', '010': '000', '011': '110',
          '100': '000', '101': '110', '110': '101', '111': '010'}]
    ])
    def test_simon(self, simon_input):
        # find the two keys that have matching values
        nbits = math.log(len(simon_input), 2)
        if math.ceil(nbits) != math.floor(nbits):
            raise AquaError('Input not the right length')
        nbits = int(nbits)
        get_key_pair = ((k1, k2) for k1, v1 in simon_input.items()
                        for k2, v2 in simon_input.items()
                        if v1 == v2 and not k1 == k2)
        try:  # matching keys found
            k1, k2 = next(get_key_pair)
            hidden = np.binary_repr(int(k1, 2) ^ int(k2, 2), nbits)
        except StopIteration as e:  # non matching keys found
            k1, k2 = None, None
            hidden = np.binary_repr(0, nbits)

        backend = get_aer_backend('qasm_simulator')
        oracle = ESOPOracle(simon_input)
        algorithm = Simon(oracle)
        result = algorithm.run(backend)
        self.assertEqual(result['result'], hidden)


if __name__ == '__main__':
    unittest.main()
