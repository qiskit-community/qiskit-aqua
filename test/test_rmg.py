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
from parameterized import parameterized
from test.common import QiskitAquaTestCase
from qiskit_aqua.utils.random_matrix_generator import *


class TestRandomMatrixGenerator(QiskitAquaTestCase):
    """Random matrix generator tests."""

    @parameterized.expand([[2], [100], [1000]])
    def test_random_unitary(self, N):
        a = random_unitary(N)
        distance = abs(np.sum(a.dot(a.T.conj()) - np.eye(N)))
        self.assertAlmostEqual(distance, 0, places=10)

    @parameterized.expand([[2], [100], [1000]])
    def test_random_hermitian(self, N):
        a = random_hermitian(N)
        distance = abs(np.sum(a-a.T.conj()))
        self.assertAlmostEqual(distance, 0, places=10)


if __name__ == '__main__':
    unittest.main()