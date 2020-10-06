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

""" Test Core Hamiltonian Symmetry Reduction """

import unittest
from test.chemistry import QiskitChemistryTestCase
import numpy as np

from qiskit import BasicAer
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.core import TransformationType, QubitMappingType
from qiskit.chemistry.qubit_transformations import FermionicTransformation
from qiskit.chemistry.ground_state_calculation import MinimumEigensolverGroundStateCalculation


class TestCoreHamiltonianSymmetries(QiskitChemistryTestCase):
    """ Core hamiltonian Driver symmetry tests. """

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

    def _validate_result(self, result, symm=True):
        self.assertAlmostEqual(result.energy, -7.882324378883, places=3)
        ref_dipole = (0.0, 0.0, -1.81741795)
        if not symm:
            np.testing.assert_almost_equal(result.dipole_moment, ref_dipole, decimal=2)
        else:
            self.assertIsNone(result.dipole_moment[0])
            self.assertIsNone(result.dipole_moment[1])
            self.assertAlmostEqual(result.dipole_moment[2], ref_dipole[2], places=2)

    def test_no_symmetry(self):
        """ No symmetry reduction """
        fermionic_transformation = FermionicTransformation\
            (transformation=TransformationType.FULL,
             qubit_mapping=QubitMappingType.JORDAN_WIGNER,
             two_qubit_reduction=False,
             freeze_core=False,
             orbital_reduction=None,
             z2symmetry_reduction=None)

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self.assertEqual(qubit_op.num_qubits, 12)
        solver = NumPyMinimumEigensolver()
        gsc = MinimumEigensolverGroundStateCalculation(fermionic_transformation, solver)
        result = gsc.compute_groundstate(self.driver)
        self._validate_result(result, False)

    def test_auto_symmetry(self):
        """ Auto symmetry reduction """
        fermionic_transformation = FermionicTransformation \
            (transformation=TransformationType.FULL,
             qubit_mapping=QubitMappingType.JORDAN_WIGNER,
             two_qubit_reduction=False,
             freeze_core=False,
             orbital_reduction=None,
             z2symmetry_reduction='auto')
        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self.assertEqual(qubit_op.num_qubits, 8)
        solver = NumPyMinimumEigensolver()
        gsc = MinimumEigensolverGroundStateCalculation(fermionic_transformation, solver)
        result = gsc.compute_groundstate(self.driver)
        self._validate_result(result)
        self.assertEqual(qubit_op.z2_symmetries.tapering_values, [1, 1, 1, 1])

    def test_given_symmetry(self):
        """ Supplied symmetry reduction """
        fermionic_transformation = FermionicTransformation\
            (transformation=TransformationType.FULL,
             qubit_mapping=QubitMappingType.JORDAN_WIGNER,
             two_qubit_reduction=False,
             freeze_core=False,
             orbital_reduction=None,
             z2symmetry_reduction=[1, 1, 1, 1])
        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self.assertEqual(qubit_op.num_qubits, 8)
        solver = NumPyMinimumEigensolver()
        gsc = MinimumEigensolverGroundStateCalculation(fermionic_transformation, solver)
        result = gsc.compute_groundstate(self.driver)
        self._validate_result(result)
        self.assertEqual(qubit_op.z2_symmetries.tapering_values, [1, 1, 1, 1])

    def test_given_symmetry_fail_len(self):
        """ Supplied symmetry reduction invalid len """
        with self.assertRaises(QiskitChemistryError):
            fermionic_transformation = FermionicTransformation\
                (transformation=TransformationType.FULL,
                 qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                 two_qubit_reduction=False,
                 freeze_core=False,
                 orbital_reduction=None,
                 z2symmetry_reduction=[1, 1, 1])

            _, _ = fermionic_transformation.transform(self.driver)


    def test_given_symmetry_fail_values(self):
        """ Supplied symmetry reduction invalid values """
        with self.assertRaises(QiskitChemistryError):
            fermionic_transformation = FermionicTransformation\
                (transformation=TransformationType.FULL,
                 qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                 two_qubit_reduction=False,
                 freeze_core=False,
                 orbital_reduction=None,
                 z2symmetry_reduction=[1, 0, 1, 1])

            _, _ = fermionic_transformation.transform(self.driver)

    def test_auto_symmetry_freeze_core(self):
        """ Auto symmetry reduction, with freeze core """
        fermionic_transformation = FermionicTransformation \
            (transformation=TransformationType.FULL,
             qubit_mapping=QubitMappingType.JORDAN_WIGNER,
             two_qubit_reduction=False,
             freeze_core=True,
             orbital_reduction=None,
             z2symmetry_reduction='auto')

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self.assertEqual(qubit_op.num_qubits, 6)
        solver = NumPyMinimumEigensolver()
        gsc = MinimumEigensolverGroundStateCalculation(fermionic_transformation, solver)
        result = gsc.compute_groundstate(self.driver)
        self._validate_result(result)
        self.assertEqual(qubit_op.z2_symmetries.tapering_values, [-1, 1, 1, -1])

    def test_auto_freeze_core_parity(self):
        """ Auto symmetry reduction, with freeze core and parity mapping """

        fermionic_transformation = FermionicTransformation\
            (transformation=TransformationType.FULL,
             qubit_mapping=QubitMappingType.PARITY,
             two_qubit_reduction=False,
             freeze_core=True,
             orbital_reduction=None,
             z2symmetry_reduction='auto')

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self.assertEqual(qubit_op.num_qubits, 6)
        solver = NumPyMinimumEigensolver()
        gsc = MinimumEigensolverGroundStateCalculation(fermionic_transformation, solver)
        result = gsc.compute_groundstate(self.driver)
        self._validate_result(result)
        self.assertEqual(qubit_op.z2_symmetries.tapering_values, [-1, 1, 1, 1])

    def test_auto_freeze_core_parity_2(self):
        """ Auto symmetry reduction, with freeze core, parity and two q reduction """
        fermionic_transformation = FermionicTransformation\
            (transformation=TransformationType.FULL,
             qubit_mapping=QubitMappingType.PARITY,
             two_qubit_reduction=True,
             freeze_core=True,
             orbital_reduction=None,
             z2symmetry_reduction='auto')

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self.assertEqual(qubit_op.num_qubits, 6)
        solver = NumPyMinimumEigensolver()
        gsc = MinimumEigensolverGroundStateCalculation(fermionic_transformation, solver)
        result = gsc.compute_groundstate(self.driver)
        self._validate_result(result)
        self.assertEqual(qubit_op.z2_symmetries.tapering_values, [1, 1])

    def test_auto_ph_freeze_core_parity_2(self):
        """ Auto symmetry reduction, with freeze core, parity and two q reduction """
        fermionic_transformation = FermionicTransformation \
            (transformation=TransformationType.PARTICLE_HOLE,
             qubit_mapping=QubitMappingType.PARITY,
             two_qubit_reduction=True,
             freeze_core=True,
             orbital_reduction=None,
             z2symmetry_reduction='auto')

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self.assertEqual(qubit_op.num_qubits, 6)
        solver = NumPyMinimumEigensolver()
        gsc = MinimumEigensolverGroundStateCalculation(fermionic_transformation, solver)
        result = gsc.compute_groundstate(self.driver)
        self._validate_result(result)
        self.assertEqual(qubit_op.z2_symmetries.tapering_values, [1, 1])

    def test_vqe_auto_symmetry_freeze_core(self):
        """ Auto symmetry reduction, with freeze core using VQE """
        fermionic_transformation = FermionicTransformation\
            (transformation=TransformationType.FULL,
             qubit_mapping=QubitMappingType.JORDAN_WIGNER,
             two_qubit_reduction=False,
             freeze_core=True,
             orbital_reduction=None,
             z2symmetry_reduction='auto')

        qubit_op, _ = fermionic_transformation.transform(self.driver)
        self.assertEqual(qubit_op.num_qubits, 6)
        num_orbitals = fermionic_transformation._molecule_info['num_orbitals']
        num_particles = fermionic_transformation._molecule_info['num_particles']
        qubit_mapping = 'jordan_wigner'
        two_qubit_reduction = fermionic_transformation._two_qubit_reduction
        z2_symmetries = qubit_op.z2_symmetries
        initial_state = HartreeFock(num_orbitals, num_particles,
                                    qubit_mapping, two_qubit_reduction, z2_symmetries.sq_list)
        var_form = UCCSD(num_orbitals=num_orbitals,
                         num_particles=num_particles,
                         initial_state=initial_state,
                         qubit_mapping=qubit_mapping,
                         two_qubit_reduction=two_qubit_reduction,
                         z2_symmetries=z2_symmetries)

        solver = VQE(var_form=var_form, optimizer=SLSQP(maxiter=500),
                     quantum_instance = BasicAer.get_backend('statevector_simulator'))
        gsc = MinimumEigensolverGroundStateCalculation(fermionic_transformation, solver)
        result = gsc.compute_groundstate(self.driver)
        self._validate_result(result)
        self.assertEqual(qubit_op.z2_symmetries.tapering_values, [-1, 1, 1, -1])


if __name__ == '__main__':
    unittest.main()
