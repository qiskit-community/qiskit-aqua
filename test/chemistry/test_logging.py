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
from test.chemistry import QiskitChemistryTestCase
from qiskit.aqua import QiskitLogDomains
from qiskit.chemistry import get_qiskit_chemistry_logging, set_qiskit_chemistry_logging
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import QiskitChemistryError


class TestLogging(QiskitChemistryTestCase):
    """test logging methods"""

    def setUp(self):
        super().setUp()
        self.current_level = get_qiskit_chemistry_logging()
        set_qiskit_chemistry_logging(logging.INFO)

    def tearDown(self):
        set_qiskit_chemistry_logging(self.current_level)
        super().tearDown()

    def test_logging_emit(self):
        """ logging emit test """
        with self.assertLogs(QiskitLogDomains.DOMAIN_CHEMISTRY.value, level='INFO') as log:
            try:
                driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                                     unit=UnitsType.ANGSTROM,
                                     basis='sto3g')
            except QiskitChemistryError:
                self.skipTest('PYSCF driver does not appear to be installed')
                return

            _ = driver.run()
            self.assertIn('PySCF', log.output[0])


if __name__ == '__main__':
    unittest.main()
