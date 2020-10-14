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

""" Test Gaussian Log Driver """

import unittest

from test.chemistry import QiskitChemistryTestCase

from qiskit.chemistry.drivers import GaussianLogDriver, GaussianLogResult
from qiskit.chemistry import QiskitChemistryError


class TestDriverGaussianLog(QiskitChemistryTestCase):
    """Gaussian Log Driver tests."""

    def setUp(self):
        super().setUp()
        self.logfile = self.get_resource_path('test_driver_gaussian_log.txt')

    def test_log_driver(self):
        """ Test the driver itself creates log and we can get a result """
        try:
            driver = GaussianLogDriver(
                ['#p B3LYP/6-31g Freq=(Anharm) Int=Ultrafine SCF=VeryTight',
                 '',
                 'CO2 geometry optimization B3LYP/cc-pVTZ',
                 '',
                 '0 1',
                 'C  -0.848629  2.067624  0.160992',
                 'O   0.098816  2.655801 -0.159738',
                 'O  -1.796073  1.479446  0.481721',
                 '',
                 ''
                 ])
            result = driver.run()
            qfc = result.quadratic_force_constants
            expected = [('1', '1', 1409.20235, 1.17003, 0.07515),
                        ('2', '2', 2526.46159, 3.76076, 0.24156),
                        ('3a', '3a', 462.61566, 0.12609, 0.0081),
                        ('3b', '3b', 462.61566, 0.12609, 0.0081)]
            self.assertListEqual(qfc, expected)
        except QiskitChemistryError:
            self.skipTest('GAUSSIAN driver does not appear to be installed')

    # These tests check the gaussian log result and the parsing from a partial log file that is
    # located with the tests so that this aspect of the code can be tested independent of
    # Gaussian 16 being installed.

    def test_gaussian_log_result_file(self):
        """ Test result from file """
        result = GaussianLogResult(self.logfile)
        with open(self.logfile) as file:
            lines = file.read().split('\n')

        with self.subTest('Check list of lines'):
            self.assertListEqual(result.log, lines)

        with self.subTest('Check as string'):
            line = '\n'.join(lines)
            self.assertEqual(str(result), line)

    def test_gaussian_log_result_list(self):
        """ Test result from list of strings """
        with open(self.logfile) as file:
            lines = file.read().split('\n')
        result = GaussianLogResult(lines)
        self.assertListEqual(result.log, lines)

    def test_gaussian_log_result_string(self):
        """ Test result from string """
        with open(self.logfile) as file:
            line = file.read()
        result = GaussianLogResult(line)
        self.assertListEqual(result.log, line.split('\n'))

    def test_quadratic_force_constants(self):
        """ Test quadratic force constants """
        result = GaussianLogResult(self.logfile)
        qfc = result.quadratic_force_constants
        expected = [('1', '1', 1409.20235, 1.17003, 0.07515),
                    ('2', '2', 2526.46159, 3.76076, 0.24156),
                    ('3a', '3a', 462.61566, 0.12609, 0.0081),
                    ('3b', '3b', 462.61566, 0.12609, 0.0081)]
        self.assertListEqual(qfc, expected)

    def test_cubic_force_constants(self):
        """ Test cubic force constants """
        result = GaussianLogResult(self.logfile)
        cfc = result.cubic_force_constants
        expected = [('1', '1', '1', -260.36071, -1.39757, -0.0475),
                    ('2', '2', '1', -498.9444, -4.80163, -0.1632),
                    ('3a', '3a', '1', 239.87769, 0.4227, 0.01437),
                    ('3a', '3b', '1', 74.25095, 0.13084, 0.00445),
                    ('3b', '3b', '1', 12.93985, 0.0228, 0.00078)]
        self.assertListEqual(cfc, expected)

    def test_quartic_force_constants(self):
        """ Test quartic force constants """
        result = GaussianLogResult(self.logfile)
        qfc = result.quartic_force_constants
        expected = [('1', '1', '1', '1', 40.39063, 1.40169, 0.02521),
                    ('2', '2', '1', '1', 79.08068, 4.92017, 0.0885),
                    ('2', '2', '2', '2', 154.78015, 17.26491, 0.31053),
                    ('3a', '3a', '1', '1', -67.10879, -0.76453, -0.01375),
                    ('3b', '3b', '1', '1', -67.10879, -0.76453, -0.01375),
                    ('3a', '3a', '2', '2', -163.29426, -3.33524, -0.05999),
                    ('3b', '3b', '2', '2', -163.29426, -3.33524, -0.05999),
                    ('3a', '3a', '3a', '3a', 220.54851, 0.82484, 0.01484),
                    ('3a', '3a', '3a', '3b', 66.77089, 0.24972, 0.00449),
                    ('3a', '3a', '3b', '3b', 117.26759, 0.43857, 0.00789),
                    ('3a', '3b', '3b', '3b', -66.77088, -0.24972, -0.00449),
                    ('3b', '3b', '3b', '3b', 220.54851, 0.82484, 0.01484)]
        self.assertListEqual(qfc, expected)

    def test_watson_hamiltonian(self):
        """ Test the watson hamiltonian """
        result = GaussianLogResult(self.logfile)
        watson = result.get_watson_hamiltonian()
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
