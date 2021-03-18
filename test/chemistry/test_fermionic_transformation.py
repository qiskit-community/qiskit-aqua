# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Fermionic Transformation """

import unittest

from test.chemistry import QiskitChemistryTestCase
from qiskit.aqua.operators import OperatorBase, I, Z
from qiskit.chemistry import QiskitChemistryError, FermionicOperator
from qiskit.chemistry.core import TransformationType, QubitMappingType
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.transformations import FermionicTransformation


class TestFermionicTransformation(QiskitChemistryTestCase):
    """Fermionic Transformation tests."""

    def setUp(self):
        super().setUp()
        try:
            driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                                 unit=UnitsType.ANGSTROM,
                                 charge=0,
                                 spin=0,
                                 basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')
        self.driver = driver

    def _validate_vars(self, fermionic_transformation, energy_shift=0.0, ph_energy_shift=0.0):
        self.assertAlmostEqual(fermionic_transformation._hf_energy, -1.117, places=3)
        self.assertAlmostEqual(fermionic_transformation._energy_shift, energy_shift)
        self.assertAlmostEqual(fermionic_transformation._ph_energy_shift, ph_energy_shift)

    def _validate_info(self, fermionic_transformation, num_particles=None,
                       num_orbitals=4, actual_two_qubit_reduction=False):
        num_particles = num_particles if num_particles is not None else (1, 1)
        z2symmetries = fermionic_transformation.molecule_info.pop('z2_symmetries')
        self.assertEqual(z2symmetries.is_empty(), True)
        self.assertEqual(fermionic_transformation.molecule_info,
                         {'num_particles': num_particles,
                          'num_orbitals': num_orbitals,
                          'two_qubit_reduction': actual_two_qubit_reduction})

    def _validate_input_object(self, qubit_op, num_qubits=4, num_paulis=15):
        self.assertTrue(isinstance(qubit_op, OperatorBase))
        self.assertIsNotNone(qubit_op)
        self.assertEqual(qubit_op.num_qubits, num_qubits)
        self.assertEqual(len(qubit_op.oplist), num_paulis)

    def test_output(self):
        """ output test """
        fermionic_transformation = FermionicTransformation(
            transformation=TransformationType.FULL,
            qubit_mapping=QubitMappingType.PARITY,
            two_qubit_reduction=True,
            freeze_core=False,
            orbital_reduction=[])

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self._validate_vars(fermionic_transformation)
        self._validate_info(fermionic_transformation, actual_two_qubit_reduction=True)
        self._validate_input_object(qubit_op, num_qubits=2, num_paulis=5)

    def test_jordan_wigner(self):
        """ jordan wigner test """
        fermionic_transformation = FermionicTransformation(
            transformation=TransformationType.FULL,
            qubit_mapping=QubitMappingType.JORDAN_WIGNER,
            two_qubit_reduction=False,
            freeze_core=False,
            orbital_reduction=[])

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self._validate_vars(fermionic_transformation)
        self._validate_info(fermionic_transformation)
        self._validate_input_object(qubit_op)

    def test_jordan_wigner_2q(self):
        """ jordan wigner 2q test """
        fermionic_transformation = FermionicTransformation(
            transformation=TransformationType.FULL,
            qubit_mapping=QubitMappingType.JORDAN_WIGNER,
            two_qubit_reduction=True,
            freeze_core=False,
            orbital_reduction=[])

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self._validate_vars(fermionic_transformation)
        # Reported effective 2 qubit reduction should be false
        self._validate_info(fermionic_transformation, actual_two_qubit_reduction=False)
        self._validate_input_object(qubit_op)

    def test_parity(self):
        """ parity test """
        fermionic_transformation = FermionicTransformation(
            transformation=TransformationType.FULL,
            qubit_mapping=QubitMappingType.PARITY,
            two_qubit_reduction=False,
            freeze_core=False,
            orbital_reduction=[])

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self._validate_vars(fermionic_transformation)
        self._validate_info(fermionic_transformation)
        self._validate_input_object(qubit_op)

    def test_bravyi_kitaev(self):
        """ bravyi kitaev test """
        fermionic_transformation = FermionicTransformation(
            transformation=TransformationType.FULL,
            qubit_mapping=QubitMappingType.BRAVYI_KITAEV,
            two_qubit_reduction=False,
            freeze_core=False,
            orbital_reduction=[])

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self._validate_vars(fermionic_transformation)
        self._validate_info(fermionic_transformation)
        self._validate_input_object(qubit_op)

    def test_particle_hole(self):
        """ particle hole test """
        fermionic_transformation = FermionicTransformation(
            transformation=TransformationType.PARTICLE_HOLE,
            qubit_mapping=QubitMappingType.JORDAN_WIGNER,
            two_qubit_reduction=False,
            freeze_core=False,
            orbital_reduction=[])

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self._validate_vars(fermionic_transformation, ph_energy_shift=-1.83696799)
        self._validate_info(fermionic_transformation)
        self._validate_input_object(qubit_op)

    def test_freeze_core(self):
        """ freeze core test -- Should be in effect a no-op for H2 """
        fermionic_transformation = FermionicTransformation(
            transformation=TransformationType.FULL,
            qubit_mapping=QubitMappingType.JORDAN_WIGNER,
            two_qubit_reduction=False,
            freeze_core=True,
            orbital_reduction=[])

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self._validate_vars(fermionic_transformation)
        self._validate_info(fermionic_transformation)
        self._validate_input_object(qubit_op)

    def test_orbital_reduction(self):
        """ orbital reduction test --- Remove virtual orbital just
            for test purposes (not sensible!)
        """
        fermionic_transformation = FermionicTransformation(
            transformation=TransformationType.FULL,
            qubit_mapping=QubitMappingType.JORDAN_WIGNER,
            two_qubit_reduction=False,
            freeze_core=False,
            orbital_reduction=[-1])

        # get dummy aux operator
        qmolecule = self.driver.run()
        fer_op = FermionicOperator(h1=qmolecule.one_body_integrals, h2=qmolecule.two_body_integrals)
        dummy = fer_op.total_particle_number()
        expected = (I ^ I) - 0.5 * (I ^ Z) - 0.5 * (Z ^ I)

        qubit_op, aux_ops = fermionic_transformation.transform(self.driver, [dummy])
        self._validate_vars(fermionic_transformation)
        self._validate_info(fermionic_transformation, num_orbitals=2)
        self._validate_input_object(qubit_op, num_qubits=2, num_paulis=4)

        # the first six aux_ops are added automatically, ours is the 7th one
        self.assertEqual(aux_ops[6], expected)


if __name__ == '__main__':
    unittest.main()
