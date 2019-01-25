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
from qiskit.chemistry.drivers import HDF5Driver
from test.test_driver import TestDriver


class TestDriverHDF5(QiskitAquaChemistryTestCase, TestDriver):
    """HDF5 Driver tests."""

    def setUp(self):
        driver = HDF5Driver(hdf5_input=self._get_resource_path('test_driver_hdf5.hdf5'))
        self.qmolecule = driver.run()


if __name__ == '__main__':
    unittest.main()
