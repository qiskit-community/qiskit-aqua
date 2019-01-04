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
from qiskit_chemistry.drivers import PyQuanteDriver, UnitsType, BasisType
from test.test_driver import TestDriver


class TestDriverPyQuante(QiskitAquaChemistryTestCase, TestDriver):
    """PYQUANTE Driver tests."""

    def setUp(self):
        try:
            driver = PyQuanteDriver(atoms='H .0 .0 .0; H .0 .0 0.735',
                                    units=UnitsType.ANGSTROM,
                                    charge=0,
                                    multiplicity=1,
                                    basis=BasisType.BSTO3G)
        except QiskitChemistryError:
            self.skipTest('PYQUANTE driver does not appear to be installed')
        self.qmolecule = driver.run()


if __name__ == '__main__':
    unittest.main()
