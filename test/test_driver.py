# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np


class TestDriver(object):
    """Common driver tests. For H2 @ 0.735, sto3g"""

    def test_driver_hf_energy(self):
        self.log.debug('QMolecule HF energy: {}'.format(self.qmolecule._hf_energy))
        self.assertAlmostEqual(self.qmolecule._hf_energy, -1.117, places=3)

    def test_driver_nuclear_repulsion_energy(self):
        self.log.debug('QMolecule Nuclear repulsion energy: {}'.format(self.qmolecule._nuclear_repulsion_energy))
        self.assertAlmostEqual(self.qmolecule._nuclear_repulsion_energy, 0.72, places=2)

    def test_driver_num_orbitals(self):
        self.log.debug('QMolecule Number of orbitals is {}'.format(self.qmolecule._num_orbitals))
        self.assertEqual(self.qmolecule._num_orbitals, 2)

    def test_driver_num_alpha(self):
        self.log.debug('QMolecule Number of alpha electrons is {}'.format(self.qmolecule._num_alpha))
        self.assertEqual(self.qmolecule._num_alpha, 1)

    def test_driver_num_beta(self):
        self.log.debug('QMolecule Number of beta electrons is {}'.format(self.qmolecule._num_beta))
        self.assertEqual(self.qmolecule._num_beta, 1)

    def test_driver_molecular_charge(self):
        self.log.debug('QMolecule molecular charge is {}'.format(self.qmolecule._molecular_charge))
        self.assertEqual(self.qmolecule._molecular_charge, 0)

    def test_driver_multiplicity(self):
        self.log.debug('QMolecule multiplicity is {}'.format(self.qmolecule._multiplicity))
        self.assertEqual(self.qmolecule._multiplicity, 1)

    def test_driver_num_atoms(self):
        self.log.debug('QMolecule num atoms {}'.format(self.qmolecule._num_atoms))
        self.assertEqual(self.qmolecule._num_atoms, 2)

    def test_driver_atom_symbol(self):
        self.log.debug('QMolecule atom symbol {}'.format(self.qmolecule._atom_symbol))
        self.assertSequenceEqual(self.qmolecule._atom_symbol, ['H', 'H'])

    def test_driver_atom_xyz(self):
        self.log.debug('QMolecule atom xyz {}'.format(self.qmolecule._atom_xyz))
        np.testing.assert_array_almost_equal(self.qmolecule._atom_xyz,
                                             [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3889]], decimal=4)

    def test_driver_mo_coeff(self):
        self.log.debug('QMolecule MO coeffs xyz {}'.format(self.qmolecule._mo_coeff))
        self.assertEqual(self.qmolecule._mo_coeff.shape, (2, 2))
        np.testing.assert_array_almost_equal(np.absolute(self.qmolecule._mo_coeff),
                                             [[0.5483, 1.2183], [0.5483, 1.2183]], decimal=4)

    def test_driver_orbital_energies(self):
        self.log.debug('QMolecule orbital energies {}'.format(self.qmolecule._orbital_energies))
        np.testing.assert_array_almost_equal(self.qmolecule._orbital_energies,
                                             [-0.5806, 0.6763], decimal=4)

    def test_driver_mo_onee_ints(self):
        self.log.debug('QMolecule MO one electron integrals {}'.format(self.qmolecule._mo_onee_ints))
        self.assertEqual(self.qmolecule._mo_onee_ints.shape, (2, 2))
        np.testing.assert_array_almost_equal(np.absolute(self.qmolecule._mo_onee_ints),
                                             [[1.2563, 0.0], [0.0, 0.4719]], decimal=4)

    def test_driver_mo_eri_ints(self):
        self.log.debug('QMolecule MO two electron integrals {}'.format(self.qmolecule._mo_eri_ints))
        self.assertEqual(self.qmolecule._mo_eri_ints.shape, (2, 2, 2, 2))
        np.testing.assert_array_almost_equal(np.absolute(self.qmolecule._mo_eri_ints),
                                             [[[[0.6757, 0.0], [0.0, 0.6646]],
                                               [[0.0, 0.1809], [0.1809, 0.0]]],
                                              [[[0.0, 0.1809], [0.1809, 0.0]],
                                               [[0.6646, 0.0], [0.0, 0.6986]]]], decimal=4)

    def test_driver_dipole_integrals(self):
        self.log.debug('QMolecule has dipole integrals {}'.format(self.qmolecule.has_dipole_integrals()))
        if self.qmolecule.has_dipole_integrals():
            self.assertEqual(self.qmolecule._x_dip_mo_ints.shape, (2, 2))
            self.assertEqual(self.qmolecule._y_dip_mo_ints.shape, (2, 2))
            self.assertEqual(self.qmolecule._z_dip_mo_ints.shape, (2, 2))
            np.testing.assert_array_almost_equal(np.absolute(self.qmolecule._x_dip_mo_ints),
                                                 [[0.0, 0.0], [0.0, 0.0]], decimal=4)
            np.testing.assert_array_almost_equal(np.absolute(self.qmolecule._y_dip_mo_ints),
                                                 [[0.0, 0.0], [0.0, 0.0]], decimal=4)
            np.testing.assert_array_almost_equal(np.absolute(self.qmolecule._z_dip_mo_ints),
                                                 [[0.6945, 0.9278], [0.9278, 0.6945]], decimal=4)
            np.testing.assert_array_almost_equal(np.absolute(self.qmolecule._nuclear_dipole_moment),
                                                 [0.0, 0.0, 1.3889], decimal=4)
