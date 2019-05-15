# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest

import numpy as np
from parameterized import parameterized
from test.common import QiskitAquaTestCase
from qiskit.aqua.utils.random_matrix_generator import random_unitary, random_hermitian


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
