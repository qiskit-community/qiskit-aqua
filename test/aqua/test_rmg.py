# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Random Matrix Generator """

import unittest
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, idata, unpack
from qiskit.aqua.utils.random_matrix_generator import random_unitary, random_hermitian


@ddt
class TestRandomMatrixGenerator(QiskitAquaTestCase):
    """Random matrix generator tests."""

    @idata([[2], [100], [1000]])
    @unpack
    def test_random_unitary(self, m_v):
        """ random unitary test """
        r_a = random_unitary(m_v)
        distance = abs(np.sum(r_a.dot(r_a.T.conj()) - np.eye(m_v)))
        self.assertAlmostEqual(distance, 0, places=10)

    @idata([[2], [100], [1000]])
    @unpack
    def test_random_hermitian(self, m_v):
        """ random hermitian test """
        r_a = random_hermitian(m_v)
        distance = abs(np.sum(r_a - r_a.T.conj()))
        self.assertAlmostEqual(distance, 0, places=10)


if __name__ == '__main__':
    unittest.main()
