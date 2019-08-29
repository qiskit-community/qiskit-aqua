# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Driver """

from abc import ABC, abstractmethod
import numpy as np


class TestDriver(ABC):
    """Common driver tests. For H2 @ 0.735, sto3g"""

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

    def test_driver_hf_energy(self):
        """ driver hf energy test """
        self.log.debug('QMolecule HF energy: {}'.format(self.qmolecule.hf_energy))
        self.assertAlmostEqual(self.qmolecule.hf_energy, -1.117, places=3)

    def test_driver_nuclear_repulsion_energy(self):
        """ driver nuclear repulsion energy test """
        self.log.debug('QMolecule Nuclear repulsion energy: {}'.format(
                        self.qmolecule.nuclear_repulsion_energy))
        self.assertAlmostEqual(self.qmolecule.nuclear_repulsion_energy, 0.72, places=2)

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

    def test_driver_molecular_charge(self):
        """ driver molecular charge test """
        self.log.debug('QMolecule molecular charge is {}'.format(self.qmolecule.molecular_charge))
        self.assertEqual(self.qmolecule.molecular_charge, 0)

    def test_driver_multiplicity(self):
        """ driver multiplicity test """
        self.log.debug('QMolecule multiplicity is {}'.format(self.qmolecule.multiplicity))
        self.assertEqual(self.qmolecule.multiplicity, 1)

    def test_driver_num_atoms(self):
        """ driver num atoms test """
        self.log.debug('QMolecule num atoms {}'.format(self.qmolecule.num_atoms))
        self.assertEqual(self.qmolecule.num_atoms, 2)

    def test_driver_atom_symbol(self):
        """ driver atom symbol test """
        self.log.debug('QMolecule atom symbol {}'.format(self.qmolecule.atom_symbol))
        self.assertSequenceEqual(self.qmolecule.atom_symbol, ['H', 'H'])

    def test_driver_atom_xyz(self):
        """ driver atom xyz test """
        self.log.debug('QMolecule atom xyz {}'.format(self.qmolecule.atom_xyz))
        np.testing.assert_array_almost_equal(self.qmolecule.atom_xyz,
                                             [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3889]], decimal=4)

    def test_driver_mo_coeff(self):
        """ driver mo coeff test """
        self.log.debug('QMolecule MO coeffs xyz {}'.format(self.qmolecule.mo_coeff))
        self.assertEqual(self.qmolecule.mo_coeff.shape, (2, 2))
        np.testing.assert_array_almost_equal(np.absolute(self.qmolecule.mo_coeff),
                                             [[0.5483, 1.2183], [0.5483, 1.2183]], decimal=4)

    def test_driver_orbital_energies(self):
        """ driver orbital energies test """
        self.log.debug('QMolecule orbital energies {}'.format(self.qmolecule.orbital_energies))
        np.testing.assert_array_almost_equal(self.qmolecule.orbital_energies,
                                             [-0.5806, 0.6763], decimal=4)

    def test_driver_mo_onee_ints(self):
        """ driver mo oneee ints test """
        self.log.debug('QMolecule MO one electron integrals {}'.format(self.qmolecule.mo_onee_ints))
        self.assertEqual(self.qmolecule.mo_onee_ints.shape, (2, 2))
        np.testing.assert_array_almost_equal(np.absolute(self.qmolecule.mo_onee_ints),
                                             [[1.2563, 0.0], [0.0, 0.4719]], decimal=4)

    def test_driver_mo_eri_ints(self):
        """ driver mo eri ints test """
        self.log.debug('QMolecule MO two electron integrals {}'.format(self.qmolecule.mo_eri_ints))
        self.assertEqual(self.qmolecule.mo_eri_ints.shape, (2, 2, 2, 2))
        np.testing.assert_array_almost_equal(np.absolute(self.qmolecule.mo_eri_ints),
                                             [[[[0.6757, 0.0], [0.0, 0.6646]],
                                               [[0.0, 0.1809], [0.1809, 0.0]]],
                                              [[[0.0, 0.1809], [0.1809, 0.0]],
                                               [[0.6646, 0.0], [0.0, 0.6986]]]], decimal=4)

    def test_driver_dipole_integrals(self):
        """ driver dipole integrals test """
        self.log.debug('QMolecule has dipole integrals {}'.format(
                        self.qmolecule.has_dipole_integrals()))
        if self.qmolecule.has_dipole_integrals():
            self.assertEqual(self.qmolecule.x_dip_mo_ints.shape, (2, 2))
            self.assertEqual(self.qmolecule.y_dip_mo_ints.shape, (2, 2))
            self.assertEqual(self.qmolecule.z_dip_mo_ints.shape, (2, 2))
            np.testing.assert_array_almost_equal(np.absolute(self.qmolecule.x_dip_mo_ints),
                                                 [[0.0, 0.0], [0.0, 0.0]], decimal=4)
            np.testing.assert_array_almost_equal(np.absolute(self.qmolecule.y_dip_mo_ints),
                                                 [[0.0, 0.0], [0.0, 0.0]], decimal=4)
            np.testing.assert_array_almost_equal(np.absolute(self.qmolecule.z_dip_mo_ints),
                                                 [[0.6945, 0.9278], [0.9278, 0.6945]], decimal=4)
            np.testing.assert_array_almost_equal(np.absolute(self.qmolecule.nuclear_dipole_moment),
                                                 [0.0, 0.0, 1.3889], decimal=4)
