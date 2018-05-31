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
from collections import OrderedDict

from test.common import QISKitAcquaChemistryTestCase
from qiskit_acqua_chemistry.drivers import ConfigurationManager
from test.test_driver import TestDriver


class TestDriverPySCF(QISKitAcquaChemistryTestCase, TestDriver):
    """PYSCF Driver tests."""

    def setUp(self):
        cfg_mgr = ConfigurationManager()
        pyscf_cfg = OrderedDict([
            ('atom', 'H .0 .0 .0; H .0 .0 0.735'),
            ('unit', 'Angstrom'),
            ('charge', 0),
            ('spin', 0),
            ('basis', 'sto3g')
        ])
        section = {'properties': pyscf_cfg}
        try:
            driver = cfg_mgr.get_driver_instance('PYSCF')
        except ModuleNotFoundError:
            self.skipTest('PYSCF driver does not appear to be installed')
        self.qmolecule = driver.run(section)


if __name__ == '__main__':
    unittest.main()
