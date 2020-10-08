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

""" Test Molecule """

import unittest
from test.chemistry import QiskitChemistryTestCase

from functools import partial
import numpy as np

from qiskit.chemistry.drivers import Molecule


class TestMolecule(QiskitChemistryTestCase):
    """Test driver-independent molecule definition."""

    def test_construct(self):
        """ test construct """
        stretch = partial(
            Molecule.absolute_stretching,
            kwargs={'atom_pair': (1, 0)})

        with self.subTest('Masses supplied'):
            mol = Molecule(geometry=[('H', [0., 0., 0.]), ('H', [0., 0., 1.])],
                           degrees_of_freedom=[stretch],
                           masses=[1, 1])
            self.assertListEqual(mol.geometry, [('H', [0., 0., 0.]), ('H', [0., 0., 1.])])
            self.assertEqual(mol.multiplicity, 1)
            self.assertEqual(mol.charge, 0)
            self.assertIsNone(mol.perturbations)
            self.assertListEqual(mol.masses, [1, 1])

        with self.subTest('No masses'):
            mol = Molecule(geometry=[('H', [0., 0., 0.]), ('H', [0., 0., 1.])],
                           degrees_of_freedom=[stretch])
            self.assertIsNone(mol.masses)

        with self.subTest('All params'):
            mol = Molecule(geometry=[('H', [0., 0., 0.]), ('H', [0., 0., 1.])],
                           multiplicity=2, charge=1,
                           degrees_of_freedom=[stretch],
                           masses=[0.7, 0.8])
            self.assertEqual(mol.multiplicity, 2)
            self.assertEqual(mol.charge, 1)
            self.assertIsNone(mol.perturbations)
            self.assertListEqual(mol.masses, [0.7, 0.8])

        with self.subTest('Mismatched masses length'):
            with self.assertRaises(ValueError):
                Molecule(geometry=[('H', [0., 0., 0.]), ('H', [0., 0., 1.])], masses=[1, 1, 1])

    def test_charge(self):
        """ test charge """
        mol = Molecule(geometry=[('H', [0., 0., 0.]), ('H', [0., 0., 1.])])
        self.assertEqual(mol.charge, 0)
        mol.charge = 1
        self.assertEqual(mol.charge, 1)

    def test_multiplicity(self):
        """ test multiplicity """
        mol = Molecule(geometry=[('H', [0., 0., 0.]), ('H', [0., 0., 1.])])
        self.assertEqual(mol.multiplicity, 1)
        mol.multiplicity = 0
        self.assertEqual(mol.multiplicity, 0)

    def test_stretch(self):
        """ test stretch """
        geom = None

        with self.subTest('From original'):
            geom = Molecule.absolute_stretching(atom_pair=(1, 0),
                                                perturbation=2,
                                                geometry=[('H', [0., 0., 0.]),
                                                          ('H', [0., 0., 1.])])
            self.assertListEqual(geom[1][1], [0., 0., 3.])

        with self.subTest('Reduce stretch'):
            geom = Molecule.absolute_stretching(atom_pair=(1, 0),
                                                perturbation=-.1,
                                                geometry=geom)
            self.assertListEqual(geom[1][1], [0., 0., 3. - .1])

    def test_bend(self):
        """ test bend """
        with self.subTest('pi/2 bend 1-0-2'):
            geom = Molecule.absolute_bending(atom_trio=(1, 0, 2),
                                             bend=np.pi / 2,
                                             geometry=[('H', [0., 0., 0.]),
                                                       ('H', [0., 0., 1.]),
                                                       ('Li', [0., 1., -1.])])
            self.assertListEqual(geom[1][1], [0., 1., 0.])

        with self.subTest('-pi/4 bend 1-0-2'):
            geom = Molecule.absolute_bending(atom_trio=(1, 0, 2),
                                             bend=-np.pi / 4,
                                             geometry=geom)
            np.testing.assert_array_almost_equal(
                geom[1][1], [0., np.sqrt(2) / 2, np.sqrt(2) / 2])

        with self.subTest('-pi/4 bend 2-0-1'):
            geom = Molecule.absolute_bending(atom_trio=(2, 0, 1),
                                             bend=-np.pi / 4,
                                             geometry=geom)
            np.testing.assert_array_almost_equal(geom[2][1], [0., 0., -np.sqrt(2)])

        # Test linear case
        with self.subTest('Linear case'):
            geom = Molecule.absolute_bending(atom_trio=(1, 0, 2),
                                             bend=np.pi / 2,
                                             geometry=[('H', [0., 0., 0.]),
                                                       ('H', [0., 0., 1.]),
                                                       ('Li', [0., 0., -1.])])
            self.assertListEqual(geom[1][1], [1., 0., 0.])

    def test_perturbations(self):
        """ test perturbations """
        stretch1 = partial(Molecule.absolute_stretching, atom_pair=(1, 0))
        bend = partial(Molecule.absolute_bending, atom_trio=(1, 0, 2))
        stretch2 = partial(Molecule.absolute_stretching, atom_pair=(0, 1))

        mol = Molecule(geometry=[('H', [0., 0., 0.]),
                                 ('O', [0., 0., 1.]),
                                 ('Li', [0., 1., -1.])],
                       degrees_of_freedom=[stretch1, bend, stretch2],
                       masses=[1, 1, 1])

        with self.subTest('Before perturbing'):
            geom = mol.geometry
            self.assertEqual(geom[0][0], 'H')
            self.assertEqual(geom[1][0], 'O')
            self.assertEqual(geom[2][0], 'Li')
            np.testing.assert_array_almost_equal(geom[0][1], [0., 0., 0.])
            np.testing.assert_array_almost_equal(geom[1][1], [0., 0., 1.])
            np.testing.assert_array_almost_equal(geom[2][1], [0., 1., -1.])
            self.assertIsNone(mol.perturbations)

        with self.subTest('Perturbations: [2, np.pi / 2, -.5]'):
            mol.perturbations = [2, np.pi / 2, -.5]
            geom = mol.geometry
            self.assertEqual(geom[0][0], 'H')
            self.assertEqual(geom[1][0], 'O')
            self.assertEqual(geom[2][0], 'Li')
            np.testing.assert_array_almost_equal(geom[0][1], [0.0, 0.5, 0.0])
            np.testing.assert_array_almost_equal(geom[1][1], [0., 3., 0.])
            np.testing.assert_array_almost_equal(geom[2][1], [0., 1., -1.])
            self.assertListEqual(mol.perturbations, [2, np.pi / 2, -.5])

        with self.subTest('Perturbations: None'):
            mol.perturbations = None  # Should be original geometry
            geom = mol.geometry
            self.assertEqual(geom[0][0], 'H')
            self.assertEqual(geom[1][0], 'O')
            self.assertEqual(geom[2][0], 'Li')
            np.testing.assert_array_almost_equal(geom[0][1], [0., 0., 0.])
            np.testing.assert_array_almost_equal(geom[1][1], [0., 0., 1.])
            np.testing.assert_array_almost_equal(geom[2][1], [0., 1., -1.])
            self.assertIsNone(mol.perturbations)


if __name__ == '__main__':
    unittest.main()
