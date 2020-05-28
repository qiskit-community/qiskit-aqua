# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Driver Gaussian """

import unittest

from test.chemistry import QiskitChemistryTestCase
from test.chemistry.test_driver import TestDriver
from qiskit.chemistry.drivers import GaussianDriver
from qiskit.chemistry import QiskitChemistryError


class TestDriverGaussian(QiskitChemistryTestCase, TestDriver):
    """Gaussian Driver tests."""

    def setUp(self):
        super().setUp()
        try:
            driver = GaussianDriver(
                ['# rhf/sto-3g scf(conventional) geom=nocrowd',
                 '',
                 'h2 molecule',
                 '',
                 '0 1',
                 'H   0.0  0.0    0.0',
                 'H   0.0  0.0    0.735',
                 ''
                 ])
        except QiskitChemistryError:
            self.skipTest('GAUSSIAN driver does not appear to be installed')
        self.qmolecule = driver.run()


if __name__ == '__main__':
    unittest.main()
