import unittest
from functools import partial

import numpy as np
from qiskit.aqua.operators.weighted_pauli_operator import WeightedPauliOperator

from qiskit.chemistry.molecule import Molecule


class TestMolecule(unittest.TestCase):
    def test_construct(self):
        stretch = partial(
            Molecule.absolute_stretching,
            kwargs={'atom_pair': (1, 0)})

        m = Molecule(geometry=[['H', [0., 0., 0.]], ['H', [0., 0., 1.]]],
                     degrees_of_freedom=[stretch],
                     masses=[1, 1])

        m = Molecule(geometry=[['H', [0., 0., 0.]], ['H', [0., 0., 1.]]],
                     degrees_of_freedom=[stretch])

    def test_stretch(self):
        geom = Molecule.absolute_stretching(atom_pair=(1, 0),
                                            perturbation=2,
                                            geometry=[['H', [0., 0., 0.]], [
                                                'H', [0., 0., 1.]]]
                                            )
        self.assertListEqual(geom[1][1], [0., 0., 3.])
        geom = Molecule.absolute_stretching(atom_pair=(1, 0),
                                            perturbation=-.1,
                                            geometry=geom
                                            )
        self.assertListEqual(geom[1][1], [0., 0., 3. - .1])

    def test_bend(self):
        geom = Molecule.absolute_bending(atom_trio=(1, 0, 2),
                                         bend=np.pi / 2,
                                         geometry=[['H', [0., 0., 0.]],
                                                   ['H', [0., 0., 1.]],
                                                   ['Li', [0., 1., -1.]],
                                                   ]
                                         )
        self.assertListEqual(geom[1][1], [0., 1., 0.])
        geom = Molecule.absolute_bending(atom_trio=(1, 0, 2),
                                         bend=-np.pi / 4,
                                         geometry=geom
                                         )
        np.testing.assert_array_almost_equal(
            geom[1][1], [0., np.sqrt(2) / 2, np.sqrt(2) / 2])
        geom = Molecule.absolute_bending(atom_trio=(2, 0, 1),
                                         bend=-np.pi / 4,
                                         geometry=geom
                                         )
        np.testing.assert_array_almost_equal(geom[2][1], [0., 0., -np.sqrt(2)])

        # Test linear case
        geom = Molecule.absolute_bending(atom_trio=(1, 0, 2),
                                         bend=np.pi / 2,
                                         geometry=[['H', [0., 0., 0.]],
                                                   ['H', [0., 0., 1.]],
                                                   ['Li', [0., 0., -1.]],
                                                   ]
                                         )
        self.assertListEqual(geom[1][1], [1., 0., 0.])

    def test_get_perturbations(self):
        stretch1 = partial(Molecule.absolute_stretching, atom_pair=(1, 0))
        bend = partial(Molecule.absolute_bending, atom_trio=(1, 0, 2))
        stretch2 = partial(Molecule.absolute_stretching, atom_pair=(0, 1))

        m = Molecule(geometry=[['H', [0., 0., 0.]],
                               ['H', [0., 0., 1.]],
                               ['Li', [0., 1., -1.]],
                               ],
                     degrees_of_freedom=[stretch1, bend, stretch2],
                     masses=[1, 1, 1])
        geom = m.get_perturbed_geom([2, np.pi / 2, -.5])
        np.testing.assert_array_almost_equal(geom[0][1], [0.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(geom[1][1], [0., 3., 0.])
        np.testing.assert_array_almost_equal(geom[2][1], [0., 1., -1.])

    def test_get_hamiltonian(self):
        stretch1 = partial(Molecule.absolute_stretching, atom_pair=(1, 0))
        bend = partial(Molecule.absolute_bending, atom_trio=(1, 0, 2))
        stretch2 = partial(Molecule.absolute_stretching, atom_pair=(0, 1))

        m = Molecule(geometry=[['Mg', [0., 0., 0.]],
                               ['H', [0., 0., 1.]],
                               ['H', [0., 1., -1.]],
                               ],
                     degrees_of_freedom=[stretch1, bend, stretch2],
                     # masses=[4, 1, 1]
                     )
        ham_geom = m.get_qubitop_hamiltonian([0, 0, 0])
        ham_per = m.get_qubitop_hamiltonian([2, np.pi / 2, -.5])
        # print(ham_geom.print_details())
        self.assertIsInstance(ham_geom, WeightedPauliOperator)
        self.assertNotEqual(ham_geom, ham_per)

        # Tapering
        m = Molecule(geometry=[['H', [0., 0., 1.]],
                               ['H', [0., 1., -1.]],
                               ],
                     degrees_of_freedom=[stretch1, stretch2],
                     tapering=True)
        ham_tap = m.get_qubitop_hamiltonian([2, -.5])
        print(ham_tap.print_details())
        self.assertIsInstance(ham_tap, WeightedPauliOperator)
        self.assertEqual(ham_tap.num_qubits, 1)

    def test_get_pyscf_str(self):
        mol_str = Molecule.get_pyscf_str(geometry=[['H', [0., 0., 0.]],
                                                   ['H', [0., 0., 1.]],
                                                   ['Li', [0., 1., -1.]],
                                                   ])
        self.assertEqual(
            mol_str,
            'H 0.0, 0.0, 0.0; H 0.0, 0.0, 1.0; Li 0.0, 1.0, -1.0')


if __name__ == '__main__':
    unittest.main()
