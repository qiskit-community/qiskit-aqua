# -*- coding: utf-8 -*-

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

""" Test Driver Methods FCIDump """

from test.chemistry.test_driver_methods import TestDriverMethods
from qiskit.chemistry.drivers import FCIDumpDriver


class TestDriverMethodsFCIDump(TestDriverMethods):
    """ Driver Methods FCIDump tests """

    def test_lih(self):
        """ LiH test """
        driver = FCIDumpDriver(self.get_resource_path('test_driver_fcidump_lih.fcidump'))
        result = self._run_driver(driver, freeze_core=False)
        self._assert_energy(result, 'lih')

    def test_oh(self):
        """ OH test """
        driver = FCIDumpDriver(self.get_resource_path('test_driver_fcidump_oh.fcidump'))
        result = self._run_driver(driver, freeze_core=False)
        self._assert_energy(result, 'oh')

    def test_lih_freeze_core(self):
        """ LiH freeze core test """
        with self.assertLogs('qiskit.chemistry', level='WARNING') as log:
            driver = FCIDumpDriver(self.get_resource_path('test_driver_fcidump_lih.fcidump'))
            result = self._run_driver(driver, freeze_core=True)
            self._assert_energy(result, 'lih')
        warning = 'WARNING:qiskit.chemistry.qmolecule:' + \
                  'Missing molecule information! Returning empty core orbital list.'
        self.assertIn(warning, log.output)

    def test_oh_freeze_core(self):
        """ OH freeze core test """
        with self.assertLogs('qiskit.chemistry', level='WARNING') as log:
            driver = FCIDumpDriver(self.get_resource_path('test_driver_fcidump_oh.fcidump'))
            result = self._run_driver(driver, freeze_core=True)
            self._assert_energy(result, 'oh')
        warning = 'WARNING:qiskit.chemistry.qmolecule:' + \
                  'Missing molecule information! Returning empty core orbital list.'
        self.assertIn(warning, log.output)

    def test_lih_with_atoms(self):
        """ LiH with num_atoms test """
        driver = FCIDumpDriver(self.get_resource_path('test_driver_fcidump_lih.fcidump'),
                               atoms=['Li', 'H'])
        result = self._run_driver(driver, freeze_core=True)
        self._assert_energy(result, 'lih')

    def test_oh_with_atoms(self):
        """ OH with num_atoms test """
        driver = FCIDumpDriver(self.get_resource_path('test_driver_fcidump_oh.fcidump'),
                               atoms=['O', 'H'])
        result = self._run_driver(driver, freeze_core=True)
        self._assert_energy(result, 'oh')
