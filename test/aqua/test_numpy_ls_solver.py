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

""" Test NumPy LS solver """

import unittest
from test.aqua import QiskitAquaTestCase
import numpy as np
from qiskit.aqua.algorithms import NumPyLSsolver


class TestNumPyLSsolver(QiskitAquaTestCase):
    """ Test NumPy LS solver """
    def setUp(self):
        super().setUp()
        self.matrix = [[1, 2], [2, 1]]
        self.vector = [1, 2]

    def test_els(self):
        """ ELS test """
        algo = NumPyLSsolver(self.matrix, self.vector)
        result = algo.run()
        np.testing.assert_array_almost_equal(result.solution, [1, 0])
        np.testing.assert_array_almost_equal(result.eigvals, [3, -1])


if __name__ == '__main__':
    unittest.main()
