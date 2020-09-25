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

""" Test Driver HDF5 via saved QMolecule on new HDF5 """

import unittest
import tempfile
import os

from test.chemistry import QiskitChemistryTestCase
from test.chemistry.test_driver import TestDriver
from qiskit.chemistry.drivers import HDF5Driver


class TestDriverHDF5Save(QiskitChemistryTestCase, TestDriver):
    """ Use HDF5 Driver to test saved HDF5 from QMolecule """

    def setUp(self):
        super().setUp()
        driver = HDF5Driver(hdf5_input=self.get_resource_path('test_driver_hdf5.hdf5'))
        temp_qmolecule = driver.run()
        file, self.save_file = tempfile.mkstemp(suffix='.hdf5')
        os.close(file)
        temp_qmolecule.save(self.save_file)
        # Tests are run on self.qmolecule which is from new saved HDF5 file
        # so save is tested based on getting expected values as per original
        driver = HDF5Driver(hdf5_input=self.save_file)
        self.qmolecule = driver.run()

    def tearDown(self):
        try:
            os.remove(self.save_file)
        except OSError:
            pass


if __name__ == '__main__':
    unittest.main()
