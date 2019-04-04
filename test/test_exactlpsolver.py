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

import numpy as np

from test.common import QiskitAquaTestCase
from qiskit.aqua import Operator, run_algorithm
from qiskit.aqua.input import LinearSystemInput
from qiskit.aqua.algorithms import ExactLPsolver


class TestExactLPsolver(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()
        self.algo_input = LinearSystemInput()
        self.algo_input.matrix = [[1, 0], [0, 1]]
        self.algo_input.vector = [1, 0]

    def test_elp_via_run_algorithm(self):
        pass

    def test_elp_direct(self):
        algo = ExactLPsolver(self.algo_input.matrix, self.algo_input.vector)
        result = algo.run()
        print(result)

if __name__ == '__main__':
    unittest.main()
