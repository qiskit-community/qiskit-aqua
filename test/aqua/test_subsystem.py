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

""" Test Subsystem Utils """

import unittest
from test.aqua import QiskitAquaTestCase
from qiskit.aqua.utils.subsystem import get_subsystems_counts


class TestSubsystem(QiskitAquaTestCase):
    """Test Subsystem utils"""

    def test_get_subsystems_counts(self):
        """Test get subsystems counts"""

        complete_system_counts = {'11 010': 1, '01 011': 1, '11 011': 1}
        result = get_subsystems_counts(complete_system_counts)

        self.assertDictEqual(result[0], {'11': 2, '01': 1})
        self.assertDictEqual(result[1], {'010': 1, '011': 2})

    def test_get_subsystems_post_selected(self):
        """Test subsystems counts with a post selection condition"""
        complete_system_counts = {'11 010': 1, '01 011': 1, '11 011': 1}
        result = get_subsystems_counts(complete_system_counts, 0, '11')

        self.assertDictEqual(result[0], {'11': 2})
        self.assertDictEqual(result[1], {'010': 1, '011': 1})


if __name__ == '__main__':
    unittest.main()
