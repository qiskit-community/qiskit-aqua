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
from test.common import QiskitAquaChemistryTestCase
from qiskit_chemistry import QiskitChemistryError
from qiskit_chemistry.drivers import PySCFDriver
from test.test_driver import TestDriver


class TestDriverPySCF(QiskitAquaChemistryTestCase, TestDriver):
    """PYSCF Driver tests."""

    def setUp(self):
        try:
            driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                                 unit='Angstrom',
                                 charge=0,
                                 spin=0,
                                 basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')
        self.qmolecule = driver.run()


if __name__ == '__main__':
    unittest.main()
