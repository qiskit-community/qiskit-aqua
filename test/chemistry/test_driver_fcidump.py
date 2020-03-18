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

""" Test Driver FCIDump """

import unittest
from abc import ABC, abstractmethod
from test.chemistry import QiskitChemistryTestCase
import numpy as np
from qiskit.chemistry.drivers import FCIDumpDriver


class BaseTestDriverFCIDump(ABC):
    """FCIDump Driver base test class using H2 @ 0.735, sto3g.

    In contrast to the other driver tests this one does *not* derive from TestDriver because the
    interface is fundamentally different.
    Similar to the HDF5Driver there is also no TestDriverMethodsFCIDump class for the same reason.
    """

    def __init__(self):
        self.log = None
        self.qmolecule = None

    @abstractmethod
    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """ asset Almost Equal """
        raise Exception('Abstract method')

    @abstractmethod
    def assertEqual(self, first, second, msg=None):
        """ assert equal """
        raise Exception('Abstract method')

    @abstractmethod
    def assertSequenceEqual(self, seq1, seq2, msg=None, seq_type=None):
        """ assert Sequence Equal """
        raise Exception('Abstract method')

    def test_driver_inactive_energy(self):
        """ driver inactive energy test """
        self.log.debug('QMolecule inactive energy is {}'.format(self.qmolecule.hf_energy))
        self.assertAlmostEqual(self.qmolecule.hf_energy, 0.7199, places=3)

    def test_driver_num_orbitals(self):
        """ driver num orbitals test """
        self.log.debug('QMolecule Number of orbitals is {}'.format(self.qmolecule.num_orbitals))
        self.assertEqual(self.qmolecule.num_orbitals, 2)

    def test_driver_num_alpha(self):
        """ driver num alpha test """
        self.log.debug('QMolecule Number of alpha electrons is {}'.format(self.qmolecule.num_alpha))
        self.assertEqual(self.qmolecule.num_alpha, 1)

    def test_driver_num_beta(self):
        """ driver num beta test """
        self.log.debug('QMolecule Number of beta electrons is {}'.format(self.qmolecule.num_beta))
        self.assertEqual(self.qmolecule.num_beta, 1)

    def _test_driver_mo_onee_ints(self, mo_onee):
        self.assertEqual(mo_onee.shape, (2, 2))
        np.testing.assert_array_almost_equal(np.absolute(mo_onee),
                                             [[1.2563, 0.0], [0.0, 0.4719]], decimal=4)

    def _test_driver_mo_eri_ints(self, mo_eri):
        self.assertEqual(mo_eri.shape, (2, 2, 2, 2))
        np.testing.assert_array_almost_equal(np.absolute(mo_eri),
                                             [[[[0.6757, 0.0], [0.0, 0.6646]],
                                               [[0.0, 0.1809], [0.1809, 0.0]]],
                                              [[[0.0, 0.1809], [0.1809, 0.0]],
                                               [[0.6646, 0.0], [0.0, 0.6986]]]], decimal=4)


class TestDriverFCIDumpRHF(QiskitChemistryTestCase, BaseTestDriverFCIDump):
    """RHF FCIDump Driver tests."""

    def setUp(self):
        super().setUp()
        driver = FCIDumpDriver(self.get_resource_path('test_driver_fcidump_rhf.fcidump'))
        self.qmolecule = driver.run()

    def test_driver_mo_onee_ints(self):
        """ driver mo onee ints test """
        self.log.debug('QMolecule MO one electron integrals are {}'.format(
            self.qmolecule.mo_onee_ints))
        self._test_driver_mo_onee_ints(self.qmolecule.mo_onee_ints)

    def test_driver_mo_eri_ints(self):
        """ driver mo eri ints test """
        self.log.debug('QMolecule MO two electron integrals are {}'.format(
            self.qmolecule.mo_eri_ints))
        self._test_driver_mo_eri_ints(self.qmolecule.mo_eri_ints)


class TestDriverFCIDumpUHF(QiskitChemistryTestCase, BaseTestDriverFCIDump):
    """UHF FCIDump Driver tests."""

    def setUp(self):
        super().setUp()
        driver = FCIDumpDriver(self.get_resource_path('test_driver_fcidump_uhf.fcidump'))
        self.qmolecule = driver.run()

    def test_driver_mo_onee_ints(self):
        """ driver alpha mo onee ints test """
        self.log.debug('QMolecule MO alpha one electron integrals are {}'.format(
            self.qmolecule.mo_onee_ints))
        self._test_driver_mo_onee_ints(self.qmolecule.mo_onee_ints)

    def test_driver_mo_onee_b_ints(self):
        """ driver beta mo onee ints test """
        self.log.debug('QMolecule MO beta one electron integrals are {}'.format(
            self.qmolecule.mo_onee_ints_b))
        self._test_driver_mo_onee_ints(self.qmolecule.mo_onee_ints_b)

    def test_driver_mo_eri_ints(self):
        """ driver alpha-alpha mo eri ints test """
        self.log.debug('QMolecule MO alpha-alpha two electron integrals are {}'.format(
            self.qmolecule.mo_eri_ints))
        self._test_driver_mo_eri_ints(self.qmolecule.mo_eri_ints)

    def test_driver_mo_eri_ints_ba(self):
        """ driver beta-alpha mo eri ints test """
        self.log.debug('QMolecule MO beta-alpha two electron integrals are {}'.format(
            self.qmolecule.mo_eri_ints_ba))
        self._test_driver_mo_eri_ints(self.qmolecule.mo_eri_ints_ba)

    def test_driver_mo_eri_ints_bb(self):
        """ driver beta-beta mo eri ints test """
        self.log.debug('QMolecule MO beta-beta two electron integrals are {}'.format(
            self.qmolecule.mo_eri_ints_bb))
        self._test_driver_mo_eri_ints(self.qmolecule.mo_eri_ints_bb)


if __name__ == '__main__':
    unittest.main()
