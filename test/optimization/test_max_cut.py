# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test max_cut module."""

import unittest
import numpy as np

from qiskit.optimization.applications.ising.max_cut import max_cut_value


class TestVariable(unittest.TestCase):
    """Test Variable."""

    def test_max_cut_value(self):
        """test init"""

        w = np.array([[0, 1], [1, 0]])
        partition = np.array([0, 0])
        self.assertEqual(max_cut_value(partition, w), 0)

        partition = np.array([0, 1])
        self.assertEqual(max_cut_value(partition, w), 1)

        partition = np.array([0, 1])
        w = np.array([[0, 2], [2, 0]])
        self.assertEqual(max_cut_value(partition, w), 2)


if __name__ == "__main__":
    unittest.main()
