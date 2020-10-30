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

""" Test Core Hamiltonian Orb Reduce """

import unittest

from test.chemistry import QiskitChemistryTestCase
from qiskit.aqua.operators import OperatorBase
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.core import TransformationType, QubitMappingType
from qiskit.chemistry.transformations import FermionicTransformation


class TestFermionicTransformationOrbReduce(QiskitChemistryTestCase):
    """core/hamiltonian Driver tests."""

    def setUp(self):
        super().setUp()
        try:
            self.driver = PySCFDriver(atom='Li .0 .0 -0.8; H .0 .0 0.8',
                                      unit=UnitsType.ANGSTROM,
                                      charge=0,
                                      spin=0,
                                      basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')

    def _validate_vars(self, fermionic_transformation, energy_shift=0.0, ph_energy_shift=0.0):
        self.assertAlmostEqual(fermionic_transformation._hf_energy, -7.862, places=3)
        self.assertAlmostEqual(fermionic_transformation._energy_shift, energy_shift)
        self.assertAlmostEqual(fermionic_transformation._ph_energy_shift, ph_energy_shift)

    def _validate_info(self, fermionic_transformation, num_particles=None,
                       num_orbitals=12, actual_two_qubit_reduction=False):
        num_particles = num_particles if num_particles is not None else (2, 2)
        z2symmetries = fermionic_transformation.molecule_info.pop('z2_symmetries')
        self.assertEqual(z2symmetries.is_empty(), True)
        self.assertEqual(fermionic_transformation.molecule_info,
                         {'num_particles': num_particles,
                          'num_orbitals': num_orbitals,
                          'two_qubit_reduction': actual_two_qubit_reduction})

    def _validate_input_object(self, qubit_op, num_qubits=12, num_paulis=631):
        self.assertTrue(isinstance(qubit_op, OperatorBase))
        self.assertIsNotNone(qubit_op)
        self.assertEqual(qubit_op.num_qubits, num_qubits)
        self.assertEqual(len(qubit_op.oplist), num_paulis)

    def test_output(self):
        """ output test """
        fermionic_transformation = \
            FermionicTransformation(transformation=TransformationType.FULL,
                                    qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                                    two_qubit_reduction=False,
                                    freeze_core=False,
                                    orbital_reduction=[])

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self._validate_vars(fermionic_transformation)
        self._validate_info(fermionic_transformation)
        self._validate_input_object(qubit_op)

    def test_parity(self):
        """ parity test """
        fermionic_transformation = \
            FermionicTransformation(transformation=TransformationType.FULL,
                                    qubit_mapping=QubitMappingType.PARITY,
                                    two_qubit_reduction=True,
                                    freeze_core=False,
                                    orbital_reduction=[])

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self._validate_vars(fermionic_transformation)
        self._validate_info(fermionic_transformation, actual_two_qubit_reduction=True)
        self._validate_input_object(qubit_op, num_qubits=10)

    def test_freeze_core(self):
        """ freeze core test """
        fermionic_transformation = \
            FermionicTransformation(transformation=TransformationType.FULL,
                                    qubit_mapping=QubitMappingType.PARITY,
                                    two_qubit_reduction=False,
                                    freeze_core=True,
                                    orbital_reduction=[])

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self._validate_vars(fermionic_transformation, energy_shift=-7.7962196)
        self._validate_info(fermionic_transformation, num_particles=(1, 1), num_orbitals=10)
        self._validate_input_object(qubit_op, num_qubits=10, num_paulis=276)

    def test_freeze_core_orb_reduction(self):
        """ freeze core orb reduction test """
        fermionic_transformation = \
            FermionicTransformation(transformation=TransformationType.FULL,
                                    qubit_mapping=QubitMappingType.PARITY,
                                    two_qubit_reduction=False,
                                    freeze_core=True,
                                    orbital_reduction=[-3, -2])

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self._validate_vars(fermionic_transformation, energy_shift=-7.7962196)
        self._validate_info(fermionic_transformation, num_particles=(1, 1), num_orbitals=6)
        self._validate_input_object(qubit_op, num_qubits=6, num_paulis=118)

    def test_freeze_core_all_reduction(self):
        """ freeze core all reduction test """
        fermionic_transformation = \
            FermionicTransformation(transformation=TransformationType.FULL,
                                    qubit_mapping=QubitMappingType.PARITY,
                                    two_qubit_reduction=True,
                                    freeze_core=True,
                                    orbital_reduction=[-3, -2])

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self._validate_vars(fermionic_transformation, energy_shift=-7.7962196)
        self._validate_info(fermionic_transformation, num_particles=(1, 1), num_orbitals=6,
                            actual_two_qubit_reduction=True)
        self._validate_input_object(qubit_op, num_qubits=4, num_paulis=100)

    def test_freeze_core_all_reduction_ph(self):
        """ freeze core all reduction ph test """
        fermionic_transformation = \
            FermionicTransformation(transformation=TransformationType.PARTICLE_HOLE,
                                    qubit_mapping=QubitMappingType.PARITY,
                                    two_qubit_reduction=True,
                                    freeze_core=True,
                                    orbital_reduction=[-2, -1])

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self._validate_vars(fermionic_transformation, energy_shift=-7.7962196,
                            ph_energy_shift=-1.05785247)
        self._validate_info(fermionic_transformation, num_particles=(1, 1), num_orbitals=6,
                            actual_two_qubit_reduction=True)
        self._validate_input_object(qubit_op, num_qubits=4, num_paulis=52)


if __name__ == '__main__':
    unittest.main()
