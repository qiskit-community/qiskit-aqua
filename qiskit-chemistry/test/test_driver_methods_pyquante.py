# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.drivers import PyQuanteDriver, UnitsType, BasisType, HFMethodType
from test.test_driver_methods import TestDriverMethods


class TestDriverMethodsPyquante(TestDriverMethods):

    def setUp(self):
        super().setup()
        try:
            PyQuanteDriver(atoms=self.LIH)
        except QiskitChemistryError:
            self.skipTest('PyQuante driver does not appear to be installed')

    def test_lih_rhf(self):
        driver = PyQuanteDriver(atoms=self.LIH, units=UnitsType.ANGSTROM,
                                charge=0, multiplicity=1, basis=BasisType.BSTO3G,
                                hf_method=HFMethodType.RHF)
        result = self._run_driver(driver)
        self._assert_energy(result, 'lih')

    def test_lih_rohf(self):
        driver = PyQuanteDriver(atoms=self.LIH, units=UnitsType.ANGSTROM,
                                charge=0, multiplicity=1, basis=BasisType.BSTO3G,
                                hf_method=HFMethodType.ROHF)
        result = self._run_driver(driver)
        self._assert_energy(result, 'lih')

    def test_lih_uhf(self):
        driver = PyQuanteDriver(atoms=self.LIH, units=UnitsType.ANGSTROM,
                                charge=0, multiplicity=1, basis=BasisType.BSTO3G,
                                hf_method=HFMethodType.UHF)
        result = self._run_driver(driver)
        self._assert_energy(result, 'lih')

    def test_oh_rohf(self):
        driver = PyQuanteDriver(atoms=self.OH, units=UnitsType.ANGSTROM,
                                charge=0, multiplicity=2, basis=BasisType.BSTO3G,
                                hf_method=HFMethodType.ROHF)
        result = self._run_driver(driver)
        self._assert_energy(result, 'oh')

    def test_oh_uhf(self):
        driver = PyQuanteDriver(atoms=self.OH, units=UnitsType.ANGSTROM,
                                charge=0, multiplicity=2, basis=BasisType.BSTO3G,
                                hf_method=HFMethodType.UHF)
        result = self._run_driver(driver)
        self._assert_energy(result, 'oh')
