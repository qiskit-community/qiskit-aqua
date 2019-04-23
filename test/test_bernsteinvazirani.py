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
import itertools
import math
from parameterized import parameterized
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.oracles import TruthTableOracle
from qiskit.aqua.algorithms import BernsteinVazirani
from test.common import QiskitAquaTestCase

bitmaps = ['00111100', '01011010']
mct_modes = ['basic', 'basic-dirty-ancilla', 'advanced', 'noancilla']
optimizations = ['off', 'qm-dlx']


class TestBernsteinVazirani(QiskitAquaTestCase):
    @parameterized.expand(
        itertools.product(bitmaps, mct_modes, optimizations)
    )
    def test_bernsteinvazirani(self, bv_input, mct_mode, optimization='off'):
        nbits = int(math.log(len(bv_input), 2))
        # compute the ground-truth classically
        parameter = ""
        for i in reversed(range(nbits)):
            bit = bv_input[2 ** i]
            parameter += bit

        backend = BasicAer.get_backend('qasm_simulator')
        oracle = TruthTableOracle(bv_input, optimization=optimization, mct_mode=mct_mode)
        algorithm = BernsteinVazirani(oracle)
        quantum_instance = QuantumInstance(backend)
        result = algorithm.run(quantum_instance=quantum_instance)
        # print(result['circuit'].draw(line_length=10000))
        self.assertEqual(result['result'], parameter)


if __name__ == '__main__':
    unittest.main()
