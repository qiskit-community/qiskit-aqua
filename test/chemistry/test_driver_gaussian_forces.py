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
            self._check_driver_result(result)

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
            self._check_driver_result(result)

        except QiskitChemistryError:
            self.skipTest('GAUSSIAN driver does not appear to be installed')

    def test_driver_logfile(self):
        """ Test the driver works with logfile (Gaussian does not need to be installed) """

        driver = GaussianForcesDriver(
            logfile=self.get_resource_path('test_driver_gaussian_log.txt'))

        result = driver.run()
        self._check_driver_result(result)

    def _check_driver_result(self, watson):
        expected = [[352.3005875, 2, 2],
                    [-352.3005875, -2, -2],
                    [631.6153975, 1, 1],
                    [-631.6153975, -1, -1],
                    [115.653915, 4, 4],
                    [-115.653915, -4, -4],
                    [115.653915, 3, 3],
                    [-115.653915, -3, -3],
                    [-15.341901966295344, 2, 2, 2],
                    [-88.2017421687633, 1, 1, 2],
                    [42.40478531359112, 4, 4, 2],
                    [26.25167512727164, 4, 3, 2],
                    [2.2874639206341865, 3, 3, 2],
                    [0.4207357291666667, 2, 2, 2, 2],
                    [4.9425425, 1, 1, 2, 2],
                    [1.6122932291666665, 1, 1, 1, 1],
                    [-4.194299375, 4, 4, 2, 2],
                    [-4.194299375, 3, 3, 2, 2],
                    [-10.20589125, 4, 4, 1, 1],
                    [-10.20589125, 3, 3, 1, 1],
                    [2.2973803125, 4, 4, 4, 4],
                    [2.7821204166666664, 4, 4, 4, 3],
                    [7.329224375, 4, 4, 3, 3],
                    [-2.7821200000000004, 4, 3, 3, 3],
                    [2.2973803125, 3, 3, 3, 3]
                    ]
        for i, entry in enumerate(watson.data):
            msg = "mode[{}]={} does not match expected {}".format(i, entry, expected[i])
            self.assertAlmostEqual(entry[0], expected[i][0], msg=msg)
            self.assertListEqual(entry[1:], expected[i][1:], msg=msg)


if __name__ == '__main__':
    unittest.main()
