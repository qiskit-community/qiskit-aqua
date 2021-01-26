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

""" Test Logging Methods """

import unittest
import logging
import math
from test.aqua import QiskitAquaTestCase
from qiskit.aqua import get_qiskit_aqua_logging, set_qiskit_aqua_logging, QiskitLogDomains
from qiskit.aqua.algorithms import Shor


class TestLogging(QiskitAquaTestCase):
    """test logging methods"""

    def setUp(self):
        super().setUp()
        self.current_level = get_qiskit_aqua_logging()
        set_qiskit_aqua_logging(logging.INFO)

    def tearDown(self):
        set_qiskit_aqua_logging(self.current_level)
        super().tearDown()

    def test_logging_emit(self):
        """ logging emit test """
        with self.assertLogs(QiskitLogDomains.DOMAIN_AQUA.value, level='INFO') as log:
            _ = Shor(int(math.pow(3, 5)))
            self.assertIn('The input integer is a power:', log.output[0])


if __name__ == '__main__':
    unittest.main()
