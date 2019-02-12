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
import math
import numpy as np
from parameterized import parameterized

from qiskit.aqua.components.oracles import TruthTableOracle
from qiskit.aqua.algorithms import BernsteinVazirani
from qiskit.aqua import get_aer_backend, AquaError

from test.common import QiskitAquaTestCase


class TestBernsteinVazirani(QiskitAquaTestCase):
    @parameterized.expand([
        [{'000': '0', '001': '0', '010': '1', '011': '1',
          '100': '1', '101': '1', '110': '0', '111': '0'}],
        [{'000': '0', '001': '1', '010': '0', '011': '1',
          '100': '1', '101': '0', '110': '1', '111': '0'}]
    ])
    def test_bernsteinvazirani(self, bv_input):
        nbits = math.log(len(bv_input), 2)
        if math.ceil(nbits) != math.floor(nbits):
            raise AquaError('Input not the right length')
        nbits = int(nbits)

        # compute the ground-truth classically
        parameter = ""
        for i in range(nbits):
            bitstring = np.binary_repr(2**i, nbits)
            bit = bv_input[bitstring]
            parameter += bit

        backend = get_aer_backend('qasm_simulator')
        oracle = TruthTableOracle(bv_input)
        algorithm = BernsteinVazirani(oracle)
        result = algorithm.run(backend)
        self.assertEqual(result['result'], parameter)


if __name__ == '__main__':
    unittest.main()
