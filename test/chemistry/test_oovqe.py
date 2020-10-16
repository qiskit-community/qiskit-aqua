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

""" Test of the OOVQE ground state calculations """
import unittest
from test.chemistry import QiskitChemistryTestCase

from qiskit.chemistry.drivers import HDF5Driver
from qiskit.providers.basicaer import BasicAer
from qiskit.circuit.library import RealAmplitudes
from qiskit.aqua import QuantumInstance
from qiskit.aqua import AquaError
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.chemistry.algorithms.ground_state_solvers import OrbitalOptimizationVQE
from qiskit.chemistry.transformations import FermionicTransformation
from qiskit.chemistry.transformations.fermionic_transformation import FermionicQubitMappingType
from qiskit.chemistry.algorithms.ground_state_solvers.minimum_eigensolver_factories.\
            vqe_uccsd_factory import VQEUCCSDFactory


class TestOOVQE(QiskitChemistryTestCase):
    """ Test OOVQE Ground State Calculation. """

    def setUp(self):
        super().setUp()

        self.driver1 = HDF5Driver(hdf5_input=self.get_resource_path('test_oovqe_h4.hdf5'))
        self.driver2 = HDF5Driver(hdf5_input=self.get_resource_path('test_oovqe_lih.hdf5'))
        self.driver3 = HDF5Driver(hdf5_input=self.get_resource_path('test_oovqe_h4_uhf.hdf5'))

        self.energy1_rotation = -3.0104
        self.energy1 = -2.77  # energy of the VQE with pUCCD ansatz and LBFGSB optimizer
        self.energy2 = -7.70
        self.energy3 = -2.50
        self.initial_point1 = [0.039374, -0.47225463, -0.61891996, 0.02598386, 0.79045546,
                               -0.04134567, 0.04944946, -0.02971617, -0.00374005, 0.77542149]

        self.seed = 50

        self.optimizer = COBYLA(maxiter=1)
        self.transformation1 = \
            FermionicTransformation(qubit_mapping=FermionicQubitMappingType.JORDAN_WIGNER,
                                    two_qubit_reduction=False)
        self.transformation2 = \
            FermionicTransformation(qubit_mapping=FermionicQubitMappingType.JORDAN_WIGNER,
                                    two_qubit_reduction=False,
                                    freeze_core=True)

        self.quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                                shots=1,
                                                seed_simulator=self.seed,
                                                seed_transpiler=self.seed)

    def test_orbital_rotations(self):
        """ Test that orbital rotations are performed correctly. """

        optimizer = COBYLA(maxiter=1)
        solver = VQEUCCSDFactory(quantum_instance=self.quantum_instance,
                                 optimizer=optimizer,
                                 excitation_type='d',
                                 same_spin_doubles=False,
                                 method_doubles='pucc')

        calc = OrbitalOptimizationVQE(self.transformation1, solver, iterative_oo=False,
                                      initial_point=self.initial_point1)

        algo_result = calc.solve(self.driver1)
        self.assertAlmostEqual(algo_result.computed_electronic_energy, self.energy1_rotation, 4)

    def test_oovqe(self):
        """ Test the simultaneous optimization of orbitals and ansatz parameters with OOVQE using
        BasicAer's statevector_simulator. """

        optimizer = COBYLA(maxiter=3, rhobeg=0.01)
        solver = VQEUCCSDFactory(quantum_instance=self.quantum_instance,
                                 optimizer=optimizer,
                                 excitation_type='d',
                                 same_spin_doubles=False,
                                 method_doubles='pucc')

        calc = OrbitalOptimizationVQE(self.transformation1, solver, iterative_oo=False,
                                      initial_point=self.initial_point1)

        algo_result = calc.solve(self.driver1)
        self.assertLessEqual(algo_result.computed_electronic_energy, self.energy1, 4)

    def test_iterative_oovqe(self):
        """ Test the iterative OOVQE using BasicAer's statevector_simulator. """

        optimizer = COBYLA(maxiter=2, rhobeg=0.01)
        solver = VQEUCCSDFactory(quantum_instance=self.quantum_instance,
                                 optimizer=optimizer,
                                 excitation_type='d',
                                 same_spin_doubles=False,
                                 method_doubles='pucc')

        calc = OrbitalOptimizationVQE(self.transformation1, solver, iterative_oo=True,
                                      initial_point=self.initial_point1, iterative_oo_iterations=2)

        algo_result = calc.solve(self.driver1)
        self.assertLessEqual(algo_result.computed_electronic_energy, self.energy1)

    def test_oovqe_with_frozen_core(self):
        """ Test the OOVQE with frozen core approximation. """

        optimizer = COBYLA(maxiter=2, rhobeg=1)
        solver = VQEUCCSDFactory(quantum_instance=self.quantum_instance,
                                 optimizer=optimizer,
                                 excitation_type='d',
                                 same_spin_doubles=False,
                                 method_doubles='pucc')

        calc = OrbitalOptimizationVQE(self.transformation2, solver, iterative_oo=False)

        algo_result = calc.solve(self.driver2)
        self.assertLessEqual(algo_result.computed_electronic_energy +
                             self.transformation2._energy_shift +
                             self.transformation2._nuclear_repulsion_energy, self.energy2)

    def test_oovqe_with_unrestricted_hf(self):
        """ Test the OOVQE with unrestricted HF method. """

        optimizer = COBYLA(maxiter=2, rhobeg=0.01)
        solver = VQEUCCSDFactory(quantum_instance=self.quantum_instance,
                                 optimizer=optimizer,
                                 excitation_type='d',
                                 same_spin_doubles=False,
                                 method_doubles='pucc')

        calc = OrbitalOptimizationVQE(self.transformation1, solver, iterative_oo=False)

        algo_result = calc.solve(self.driver3)
        self.assertLessEqual(algo_result.computed_electronic_energy, self.energy3)

    def test_oovqe_with_unsupported_varform(self):
        """ Test the OOVQE with unsupported varform. """

        optimizer = COBYLA(maxiter=2, rhobeg=0.01)
        solver = VQE(var_form=RealAmplitudes(), optimizer=optimizer,
                     quantum_instance=self.quantum_instance)

        calc = OrbitalOptimizationVQE(self.transformation1, solver, iterative_oo=False)

        with self.assertRaises(AquaError):
            calc.solve(self.driver3)

    def test_oovqe_with_vqe_uccsd(self):
        """ Test the OOVQE with VQE + UCCSD instead of factory. """

        optimizer = COBYLA(maxiter=3, rhobeg=0.01)
        solver_factory = VQEUCCSDFactory(quantum_instance=self.quantum_instance,
                                         optimizer=optimizer,
                                         excitation_type='d',
                                         same_spin_doubles=False,
                                         method_doubles='pucc')
        self.transformation1.transform(self.driver1)
        solver = solver_factory.get_solver(self.transformation1)

        calc = OrbitalOptimizationVQE(self.transformation1, solver, iterative_oo=False,
                                      initial_point=self.initial_point1)

        algo_result = calc.solve(self.driver1)
        self.assertLessEqual(algo_result.computed_electronic_energy, self.energy1, 4)


if __name__ == '__main__':
    unittest.main()
