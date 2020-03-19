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
    """FCIDump Driver base test class.

    In contrast to the other driver tests this one does *not* derive from TestDriver because the
    interface is fundamentally different.
    """

    def __init__(self):
        self.log = None
        self.qmolecule = None
        self.nuclear_repulsion_energy = None
        self.num_orbitals = None
        self.num_alpha = None
        self.num_beta = None
        self.mo_onee = None
        self.mo_onee_b = None
        self.mo_eri = None
        self.mo_eri_ba = None
        self.mo_eri_bb = None

    @abstractmethod
    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """ assert Almost Equal """
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
        self.log.debug('QMolecule inactive energy is {}'.format(
            self.qmolecule.nuclear_repulsion_energy))
        self.assertAlmostEqual(self.qmolecule.nuclear_repulsion_energy,
                               self.nuclear_repulsion_energy, places=3)

    def test_driver_num_orbitals(self):
        """ driver num orbitals test """
        self.log.debug('QMolecule Number of orbitals is {}'.format(self.qmolecule.num_orbitals))
        self.assertEqual(self.qmolecule.num_orbitals, self.num_orbitals)

    def test_driver_num_alpha(self):
        """ driver num alpha test """
        self.log.debug('QMolecule Number of alpha electrons is {}'.format(self.qmolecule.num_alpha))
        self.assertEqual(self.qmolecule.num_alpha, self.num_alpha)

    def test_driver_num_beta(self):
        """ driver num beta test """
        self.log.debug('QMolecule Number of beta electrons is {}'.format(self.qmolecule.num_beta))
        self.assertEqual(self.qmolecule.num_beta, self.num_beta)

    def test_driver_mo_onee_ints(self):
        """ driver alpha mo onee ints test """
        self.log.debug('QMolecule MO alpha one electron integrals are {}'.format(
            self.qmolecule.mo_onee_ints))
        self.assertEqual(self.qmolecule.mo_onee_ints.shape, self.mo_onee.shape)
        np.testing.assert_array_almost_equal(np.absolute(self.qmolecule.mo_onee_ints),
                                             np.absolute(self.mo_onee), decimal=4)

    def test_driver_mo_onee_b_ints(self):
        """ driver beta mo onee ints test """
        if self.mo_onee_b is None:
            return
        self.log.debug('QMolecule MO beta one electron integrals are {}'.format(
            self.qmolecule.mo_onee_ints_b))
        self.assertEqual(self.qmolecule.mo_onee_ints_b.shape, self.mo_onee_b.shape)
        np.testing.assert_array_almost_equal(np.absolute(self.qmolecule.mo_onee_ints_b),
                                             np.absolute(self.mo_onee_b), decimal=4)

    def test_driver_mo_eri_ints(self):
        """ driver alpha-alpha mo eri ints test """
        self.log.debug('QMolecule MO alpha-alpha two electron integrals are {}'.format(
            self.qmolecule.mo_eri_ints))
        self.assertEqual(self.qmolecule.mo_eri_ints.shape, self.mo_eri.shape)
        np.testing.assert_array_almost_equal(np.absolute(self.qmolecule.mo_eri_ints),
                                             np.absolute(self.mo_eri), decimal=4)

    def test_driver_mo_eri_ints_ba(self):
        """ driver beta-alpha mo eri ints test """
        if self.mo_eri_ba is None:
            return
        self.log.debug('QMolecule MO beta-alpha two electron integrals are {}'.format(
            self.qmolecule.mo_eri_ints_ba))
        self.assertEqual(self.qmolecule.mo_eri_ints_ba.shape, self.mo_eri_ba.shape)
        np.testing.assert_array_almost_equal(np.absolute(self.qmolecule.mo_eri_ints_ba),
                                             np.absolute(self.mo_eri_ba), decimal=4)

    def test_driver_mo_eri_ints_bb(self):
        """ driver beta-beta mo eri ints test """
        if self.mo_eri_bb is None:
            return
        self.log.debug('QMolecule MO beta-beta two electron integrals are {}'.format(
            self.qmolecule.mo_eri_ints_bb))
        self.assertEqual(self.qmolecule.mo_eri_ints_bb.shape, self.mo_eri_bb.shape)
        np.testing.assert_array_almost_equal(np.absolute(self.qmolecule.mo_eri_ints_bb),
                                             np.absolute(self.mo_eri_bb), decimal=4)


class TestDriverFCIDumpH2(QiskitChemistryTestCase, BaseTestDriverFCIDump):
    """RHF FCIDump Driver tests."""

    def setUp(self):
        super().setUp()
        self.nuclear_repulsion_energy = 0.7199
        self.num_orbitals = 2
        self.num_alpha = 1
        self.num_beta = 1
        self.mo_onee = np.array([[1.2563, 0.0], [0.0, 0.4719]])
        self.mo_onee_b = None
        self.mo_eri = np.array([[[[0.6757, 0.0], [0.0, 0.6646]],
                                 [[0.0, 0.1809], [0.1809, 0.0]]],
                                [[[0.0, 0.1809], [0.1809, 0.0]],
                                 [[0.6646, 0.0], [0.0, 0.6986]]]])
        self.mo_eri_ba = None
        self.mo_eri_bb = None
        driver = FCIDumpDriver(self.get_resource_path('test_driver_fcidump_h2.fcidump'))
        self.qmolecule = driver.run()


class TestDriverFCIDumpLiH(QiskitChemistryTestCase, BaseTestDriverFCIDump):
    """RHF FCIDump Driver tests."""

    def setUp(self):
        super().setUp()
        self.nuclear_repulsion_energy = 0.9924
        self.num_orbitals = 6
        self.num_alpha = 2
        self.num_beta = 2
        loaded = np.load(self.get_resource_path('test_driver_fcidump_lih.npz'))
        self.mo_onee = loaded['mo_onee']
        self.mo_onee_b = None
        self.mo_eri = loaded['mo_eri']
        self.mo_eri_ba = None
        self.mo_eri_bb = None
        driver = FCIDumpDriver(self.get_resource_path('test_driver_fcidump_lih.fcidump'))
        self.qmolecule = driver.run()


class TestDriverFCIDumpOH(QiskitChemistryTestCase, BaseTestDriverFCIDump):
    """RHF FCIDump Driver tests."""

    def setUp(self):
        super().setUp()
        self.nuclear_repulsion_energy = 11.3412
        self.num_orbitals = 6
        self.num_alpha = 5
        self.num_beta = 4
        loaded = np.load(self.get_resource_path('test_driver_fcidump_oh.npz'))
        self.mo_onee = loaded['mo_onee']
        self.mo_onee_b = loaded['mo_onee_b']
        self.mo_eri = loaded['mo_eri']
        self.mo_eri_ba = loaded['mo_eri_ba']
        self.mo_eri_bb = loaded['mo_eri_bb']
        driver = FCIDumpDriver(self.get_resource_path('test_driver_fcidump_oh.fcidump'))
        self.qmolecule = driver.run()


if __name__ == '__main__':
    unittest.main()
