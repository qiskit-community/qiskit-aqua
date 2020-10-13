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

""" Test Gaussian Forces Driver """

import unittest

from test.chemistry import QiskitChemistryTestCase

from qiskit.chemistry.drivers import GaussianForcesDriver, Molecule
from qiskit.chemistry import QiskitChemistryError


class TestDriverGaussianForces(QiskitChemistryTestCase):
    """Gaussian Forces Driver tests."""

    def setUp(self):
        super().setUp()

    def test_driver_jcf(self):
        """ Test the driver works with job control file """
        try:
            driver = GaussianForcesDriver(
                ['#p B3LYP/6-31g Freq=(Anharm) Int=Ultrafine SCF=VeryTight',
                 '',
                 'CO2 geometry optimization B3LYP/6-31g',
                 '',
                 '0 1',
                 'C  -0.848629  2.067624  0.160992',
                 'O   0.098816  2.655801 -0.159738',
                 'O  -1.796073  1.479446  0.481721',
                 '',
                 ''
                 ])
            result = driver.run()
            # TODO check result

        except QiskitChemistryError:
            self.skipTest('GAUSSIAN driver does not appear to be installed')

    def test_driver_molecule(self):
        """ Test the driver works with Molecule """
        try:
            driver = GaussianForcesDriver(
                molecule=Molecule(geometry=[('C', [-0.848629, 2.067624, 0.160992]),
                                            ('O', [0.098816, 2.655801, -0.159738]),
                                            ('O', [-1.796073, 1.479446, 0.481721])],
                                  multiplicity=1,
                                  charge=0),
                basis='6-31g')
            result = driver.run()
            # TODO check result

        except QiskitChemistryError:
            self.skipTest('GAUSSIAN driver does not appear to be installed')


if __name__ == '__main__':
    unittest.main()
