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
from test.common import QiskitChemistryTestCase
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from test.test_driver import TestDriver


class TestDriverPySCF(QiskitChemistryTestCase, TestDriver):
    """PYSCF Driver tests."""

    def setUp(self):
        super().setUp()
        try:
            driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                                 unit=UnitsType.ANGSTROM,
                                 charge=0,
                                 spin=0,
                                 basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')
        self.qmolecule = driver.run()


if __name__ == '__main__':
    unittest.main()
