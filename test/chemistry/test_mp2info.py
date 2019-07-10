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

import unittest

import numpy as np

from qiskit.chemistry import QiskitChemistryError, MP2Info
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from test.common import QiskitChemistryTestCase


class TestMP2Info(QiskitChemistryTestCase):
    """Test Mp2 Info class - uses PYSCF drive to get molecule."""

    def setUp(self):
        super().setUp()
        try:
            driver = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 1.6',
                                 unit=UnitsType.ANGSTROM,
                                 charge=0,
                                 spin=0,
                                 basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')
        self.qmolecule = driver.run()
        self.mp2info = MP2Info(self.qmolecule)

    def test_mp2_delta(self):
        self.assertAlmostEqual(-0.012903900586859602, self.mp2info.mp2_delta, places=6)

    def test_mp2_energy(self):
        self.assertAlmostEqual(-7.874768670395503, self.mp2info.mp2_energy, places=6)

    def test_mp2_terms(self):
        terms = self.mp2info.mp2_terms()
        self.assertEqual(76, len(terms.keys()))

    def test_mp2_terms_frozen_core(self):
        terms = self.mp2info.mp2_terms(True)
        self.assertEqual(16, len(terms.keys()))

    def test_mp2_terms_frozen_core_orbital_reduction(self):
        terms = self.mp2info.mp2_terms(True, [-3, -2])
        self.assertEqual(4, len(terms.keys()))

    def test_mp2_get_term_info(self):
        excitations = [[0, 1, 5, 9], [0, 4, 5, 9]]
        coeffs, e_deltas = self.mp2info.mp2_get_term_info(excitations, True)
        np.testing.assert_array_almost_equal([0.028919010908783453, -0.07438748755263687],
                                             coeffs, decimal=6)
        np.testing.assert_array_almost_equal([-0.0010006159224579285, -0.009218577508137853],
                                             e_deltas, decimal=6)


if __name__ == '__main__':
    unittest.main()
