# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Driver Methods Pyquante """

import unittest

from test.chemistry.test_driver_methods_gsc import TestDriverMethods
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.drivers import PyQuanteDriver, UnitsType, BasisType, HFMethodType


class TestDriverMethodsPyquante(TestDriverMethods):
    """ Driver Methods Pyquante tests """

    def setUp(self):
        super().setUp()
        try:
            PyQuanteDriver(atoms=self.lih)
        except QiskitChemistryError:
            self.skipTest('PyQuante driver does not appear to be installed')

    def test_lih_rhf(self):
        """ lih rhf test """
        driver = PyQuanteDriver(atoms=self.lih, units=UnitsType.ANGSTROM,
                                charge=0, multiplicity=1, basis=BasisType.BSTO3G,
                                hf_method=HFMethodType.RHF)
        result = self._run_driver(driver)
        self._assert_energy(result, 'lih')

    def test_lih_rohf(self):
        """ lijh rohf test """
        driver = PyQuanteDriver(atoms=self.lih, units=UnitsType.ANGSTROM,
                                charge=0, multiplicity=1, basis=BasisType.BSTO3G,
                                hf_method=HFMethodType.ROHF)
        result = self._run_driver(driver)
        self._assert_energy(result, 'lih')

    def test_lih_uhf(self):
        """ lih uhf test """
        driver = PyQuanteDriver(atoms=self.lih, units=UnitsType.ANGSTROM,
                                charge=0, multiplicity=1, basis=BasisType.BSTO3G,
                                hf_method=HFMethodType.UHF)
        result = self._run_driver(driver)
        self._assert_energy(result, 'lih')

    def test_oh_rohf(self):
        """ oh rohf test """
        driver = PyQuanteDriver(atoms=self.o_h, units=UnitsType.ANGSTROM,
                                charge=0, multiplicity=2, basis=BasisType.BSTO3G,
                                hf_method=HFMethodType.ROHF)
        result = self._run_driver(driver)
        self._assert_energy(result, 'oh')

    def test_oh_uhf(self):
        """ oh uhf test """
        driver = PyQuanteDriver(atoms=self.o_h, units=UnitsType.ANGSTROM,
                                charge=0, multiplicity=2, basis=BasisType.BSTO3G,
                                hf_method=HFMethodType.UHF)
        result = self._run_driver(driver)
        self._assert_energy(result, 'oh')


if __name__ == '__main__':
    unittest.main()
