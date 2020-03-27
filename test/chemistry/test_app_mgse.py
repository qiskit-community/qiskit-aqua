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

""" Test molecular ground state energy application """

import unittest

from test.chemistry import QiskitChemistryTestCase
import numpy as np

from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE, IQPEMinimumEigensolver
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.aqua.components.variational_forms import RY
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.applications import MolecularGroundStateEnergy
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.core import QubitMappingType
from qiskit.chemistry.drivers import PySCFDriver, UnitsType


class TestAppMGSE(QiskitChemistryTestCase):
    """Test molecular ground state energy application """

    def setUp(self):
        super().setUp()
        try:
            self.driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                                      unit=UnitsType.ANGSTROM,
                                      charge=0,
                                      spin=0,
                                      basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')

        self.npme = NumPyMinimumEigensolver()

        self.vqe = VQE(var_form=RY(2))
        self.vqe.set_backend(BasicAer.get_backend('statevector_simulator'))

        self.reference_energy = -1.137306

    def test_mgse_npme(self):
        """ Test Molecular Ground State Energy NumPy classical solver """
        mgse = MolecularGroundStateEnergy(self.driver, self.npme)
        result = mgse.compute_energy()
        self.assertAlmostEqual(result.energy, self.reference_energy, places=5)

        formatted = result.formatted
        # Check formatted output conforms, some substrings to avoid numbers whose digits may
        # vary slightly
        self.assertEqual(len(formatted), 19)
        self.assertEqual(formatted[0], '=== GROUND STATE ENERGY ===')
        self.assertEqual(formatted[4], '  - frozen energy part: 0.0')
        self.assertEqual(formatted[5], '  - particle hole part: 0.0')
        self.assertEqual(formatted[7][0:44], '> Total ground state energy (Hartree): -1.13')
        self.assertEqual(formatted[8],
                         '  Measured:: # Particles: 2.000 S: 0.000 S^2: 0.000 M: 0.00000')
        self.assertEqual(formatted[10], '=== DIPOLE MOMENT ===')
        self.assertEqual(formatted[14], '  - frozen energy part: [0.0  0.0  0.0]')
        self.assertEqual(formatted[15], '  - particle hole part: [0.0  0.0  0.0]')
        self.assertEqual(formatted[18], '               (debye): [0.0  0.0  0.0]  Total: 0.')

    def test_mgse_vqe(self):
        """ Test Molecular Ground State Energy VQE solver """
        mgse = MolecularGroundStateEnergy(self.driver, self.vqe)
        result = mgse.compute_energy()
        self.assertAlmostEqual(result.energy, self.reference_energy, places=5)

    def test_mgse_solver(self):
        """ Test Molecular Ground State Energy setting solver """
        mgse = MolecularGroundStateEnergy(self.driver)
        with self.assertRaises(QiskitChemistryError):
            _ = mgse.compute_energy()

        mgse.solver = self.npme
        result = mgse.compute_energy()
        self.assertAlmostEqual(result.energy, self.reference_energy, places=5)

        mgse.solver = self.vqe
        result = mgse.compute_energy()
        self.assertAlmostEqual(result.energy, self.reference_energy, places=5)

    def test_mgse_callback_ipqe(self):
        """ Callback test setting up Hartree Fock with IQPE """

        def cb_create_solver(num_particles, num_orbitals,
                             qubit_mapping, two_qubit_reduction, z2_symmetries):
            state_in = HartreeFock(2, num_orbitals, num_particles, qubit_mapping,
                                   two_qubit_reduction, z2_symmetries.sq_list)
            iqpe = IQPEMinimumEigensolver(None, state_in, num_time_slices=1, num_iterations=6,
                                          expansion_mode='suzuki', expansion_order=2,
                                          shallow_circuit_concat=True)
            iqpe.quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                                    shots=100)
            return iqpe

        mgse = MolecularGroundStateEnergy(self.driver)
        result = mgse.compute_energy(cb_create_solver)
        np.testing.assert_approx_equal(result.energy, self.reference_energy, significant=2)

    def test_mgse_callback_vqe_uccsd(self):
        """ Callback test setting up Hartree Fock with UCCSD and VQE """

        def cb_create_solver(num_particles, num_orbitals,
                             qubit_mapping, two_qubit_reduction, z2_symmetries):
            initial_state = HartreeFock(2, num_orbitals, num_particles, qubit_mapping,
                                        two_qubit_reduction, z2_symmetries.sq_list)
            var_form = UCCSD(2, depth=1,
                             num_orbitals=num_orbitals,
                             num_particles=num_particles,
                             initial_state=initial_state,
                             qubit_mapping=qubit_mapping,
                             two_qubit_reduction=two_qubit_reduction,
                             z2_symmetries=z2_symmetries)
            vqe = VQE(var_form=var_form, optimizer=SLSQP(maxiter=500))
            vqe.quantum_instance = BasicAer.get_backend('statevector_simulator')
            return vqe

        mgse = MolecularGroundStateEnergy(self.driver)
        result = mgse.compute_energy(cb_create_solver)
        self.assertAlmostEqual(result.energy, self.reference_energy, places=5)

    def test_mgse_callback(self):
        """ Callback testing """
        mgse = MolecularGroundStateEnergy(self.driver)

        result = mgse.compute_energy(lambda *args: NumPyMinimumEigensolver())
        self.assertAlmostEqual(result.energy, self.reference_energy, places=5)

        result = mgse.compute_energy(lambda *args: self.vqe)
        self.assertAlmostEqual(result.energy, self.reference_energy, places=5)

    def test_mgse_default_solver(self):
        """ Callback testing using default solver """
        mgse = MolecularGroundStateEnergy(self.driver)

        result = mgse.compute_energy(mgse.get_default_solver(
            BasicAer.get_backend('statevector_simulator')))
        self.assertAlmostEqual(result.energy, self.reference_energy, places=5)

        q_inst = QuantumInstance(BasicAer.get_backend('statevector_simulator'))
        result = mgse.compute_energy(mgse.get_default_solver(q_inst))
        self.assertAlmostEqual(result.energy, self.reference_energy, places=5)

    def test_mgse_callback_vqe_uccsd_z2(self):
        """ Callback test setting up Hartree Fock with UCCSD and VQE, plus z2 symmetries """

        def cb_create_solver(num_particles, num_orbitals,
                             qubit_mapping, two_qubit_reduction, z2_symmetries):
            initial_state = HartreeFock(6, num_orbitals, num_particles, qubit_mapping,
                                        two_qubit_reduction, z2_symmetries.sq_list)
            var_form = UCCSD(6, depth=1,
                             num_orbitals=num_orbitals,
                             num_particles=num_particles,
                             initial_state=initial_state,
                             qubit_mapping=qubit_mapping,
                             two_qubit_reduction=two_qubit_reduction,
                             z2_symmetries=z2_symmetries)
            vqe = VQE(var_form=var_form, optimizer=SLSQP(maxiter=500))
            vqe.quantum_instance = BasicAer.get_backend('statevector_simulator')
            return vqe

        driver = PySCFDriver(atom='Li .0 .0 -0.8; H .0 .0 0.8')
        mgse = MolecularGroundStateEnergy(driver, qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                                          two_qubit_reduction=False, freeze_core=True,
                                          z2symmetry_reduction='auto')
        result = mgse.compute_energy(cb_create_solver)
        self.assertAlmostEqual(result.energy, -7.882, places=3)


if __name__ == '__main__':
    unittest.main()
