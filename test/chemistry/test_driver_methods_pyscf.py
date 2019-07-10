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
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, HFMethodType
from qiskit.chemistry.core import TransformationType, QubitMappingType
from .test_driver_methods import TestDriverMethods


class TestDriverMethodsPySCF(TestDriverMethods):

    def setUp(self):
        super().setup()
        try:
            PySCFDriver(atom=self.LIH)
        except QiskitChemistryError:
            self.skipTest('PySCF driver does not appear to be installed')

    def test_lih_rhf(self):
        driver = PySCFDriver(atom=self.LIH, unit=UnitsType.ANGSTROM,
                             charge=0, spin=0, basis='sto-3g',
                             hf_method=HFMethodType.RHF)
        result = self._run_driver(driver)
        self._assert_energy_and_dipole(result, 'lih')

    def test_lih_rohf(self):
        driver = PySCFDriver(atom=self.LIH, unit=UnitsType.ANGSTROM,
                             charge=0, spin=0, basis='sto-3g',
                             hf_method=HFMethodType.ROHF)
        result = self._run_driver(driver)
        self._assert_energy_and_dipole(result, 'lih')

    def test_lih_uhf(self):
        driver = PySCFDriver(atom=self.LIH, unit=UnitsType.ANGSTROM,
                             charge=0, spin=0, basis='sto-3g',
                             hf_method=HFMethodType.UHF)
        result = self._run_driver(driver)
        self._assert_energy_and_dipole(result, 'lih')

    def test_lih_rhf_parity(self):
        driver = PySCFDriver(atom=self.LIH, unit=UnitsType.ANGSTROM,
                             charge=0, spin=0, basis='sto-3g',
                             hf_method=HFMethodType.RHF)
        result = self._run_driver(driver, transformation=TransformationType.FULL,
                                  qubit_mapping=QubitMappingType.PARITY, two_qubit_reduction=False)
        self._assert_energy_and_dipole(result, 'lih')

    def test_lih_rhf_parity_2q(self):
        driver = PySCFDriver(atom=self.LIH, unit=UnitsType.ANGSTROM,
                             charge=0, spin=0, basis='sto-3g',
                             hf_method=HFMethodType.RHF)
        result = self._run_driver(driver, transformation=TransformationType.FULL,
                                  qubit_mapping=QubitMappingType.PARITY, two_qubit_reduction=True)
        self._assert_energy_and_dipole(result, 'lih')

    def test_lih_rhf_bk(self):
        driver = PySCFDriver(atom=self.LIH, unit=UnitsType.ANGSTROM,
                             charge=0, spin=0, basis='sto-3g',
                             hf_method=HFMethodType.RHF)
        result = self._run_driver(driver, transformation=TransformationType.FULL,
                                  qubit_mapping=QubitMappingType.BRAVYI_KITAEV)
        self._assert_energy_and_dipole(result, 'lih')

    def test_oh_rohf(self):
        driver = PySCFDriver(atom=self.OH, unit=UnitsType.ANGSTROM,
                             charge=0, spin=1, basis='sto-3g',
                             hf_method=HFMethodType.ROHF)
        result = self._run_driver(driver)
        self._assert_energy_and_dipole(result, 'oh')

    def test_oh_uhf(self):
        driver = PySCFDriver(atom=self.OH, unit=UnitsType.ANGSTROM,
                             charge=0, spin=1, basis='sto-3g',
                             hf_method=HFMethodType.UHF)
        result = self._run_driver(driver)
        self._assert_energy_and_dipole(result, 'oh')

    def test_oh_rohf_parity(self):
        driver = PySCFDriver(atom=self.OH, unit=UnitsType.ANGSTROM,
                             charge=0, spin=1, basis='sto-3g',
                             hf_method=HFMethodType.ROHF)
        result = self._run_driver(driver, transformation=TransformationType.FULL,
                                  qubit_mapping=QubitMappingType.PARITY, two_qubit_reduction=False)
        self._assert_energy_and_dipole(result, 'oh')

    def test_oh_rohf_parity_2q(self):
        driver = PySCFDriver(atom=self.OH, unit=UnitsType.ANGSTROM,
                             charge=0, spin=1, basis='sto-3g',
                             hf_method=HFMethodType.ROHF)
        result = self._run_driver(driver, transformation=TransformationType.FULL,
                                  qubit_mapping=QubitMappingType.PARITY, two_qubit_reduction=True)
        self._assert_energy_and_dipole(result, 'oh')

    def test_oh_uhf_parity(self):
        driver = PySCFDriver(atom=self.OH, unit=UnitsType.ANGSTROM,
                             charge=0, spin=1, basis='sto-3g',
                             hf_method=HFMethodType.UHF)
        result = self._run_driver(driver, transformation=TransformationType.FULL,
                                  qubit_mapping=QubitMappingType.PARITY, two_qubit_reduction=False)
        self._assert_energy_and_dipole(result, 'oh')

    def test_oh_uhf_parity_2q(self):
        driver = PySCFDriver(atom=self.OH, unit=UnitsType.ANGSTROM,
                             charge=0, spin=1, basis='sto-3g',
                             hf_method=HFMethodType.UHF)
        result = self._run_driver(driver, transformation=TransformationType.FULL,
                                  qubit_mapping=QubitMappingType.PARITY, two_qubit_reduction=True)
        self._assert_energy_and_dipole(result, 'oh')

    def test_oh_rohf_bk(self):
        driver = PySCFDriver(atom=self.OH, unit=UnitsType.ANGSTROM,
                             charge=0, spin=1, basis='sto-3g',
                             hf_method=HFMethodType.ROHF)
        result = self._run_driver(driver, transformation=TransformationType.FULL,
                                  qubit_mapping=QubitMappingType.BRAVYI_KITAEV)
        self._assert_energy_and_dipole(result, 'oh')

    def test_oh_uhf_bk(self):
        driver = PySCFDriver(atom=self.OH, unit=UnitsType.ANGSTROM,
                             charge=0, spin=1, basis='sto-3g',
                             hf_method=HFMethodType.UHF)
        result = self._run_driver(driver, transformation=TransformationType.FULL,
                                  qubit_mapping=QubitMappingType.BRAVYI_KITAEV)
        self._assert_energy_and_dipole(result, 'oh')
