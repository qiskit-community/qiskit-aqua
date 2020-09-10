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

""" Test OOVQE """

import logging
import unittest

from test.chemistry import QiskitChemistryTestCase

from ddt import ddt
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua import aqua_globals
from qiskit.aqua import set_qiskit_aqua_logging
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.operators.expectations import MatrixExpectation
from qiskit.chemistry.algorithms.minimum_eigen_solvers import OOVQE
from qiskit.chemistry.algorithms.minimum_eigen_solvers.oovqe import OOVQEResult
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.core import Hamiltonian
from qiskit.chemistry.core import TransformationType, QubitMappingType
from qiskit.chemistry.drivers import HDF5Driver

set_qiskit_aqua_logging(logging.INFO)


@ddt
class TestOOVQE(QiskitChemistryTestCase):
    """ Test of the OOVQE algorithm"""

    def setUp(self):

        super().setUp()
        self.energy1_rotation = -3.0104
        self.energy1 = -2.77  # energy of the VQE with pUCCD ansatz and LBFGSB optimizer
        self.energy2 = -7.70
        self.energy3 = -2.50
        self.initial_point1 = [0.039374, -0.47225463, -0.61891996, 0.02598386, 0.79045546,
                               -0.04134567, 0.04944946, -0.02971617, -0.00374005, 0.77542149]
        self.seed = 50
        aqua_globals.random_seed = self.seed
        self.quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                                shots=1,
                                                seed_simulator=self.seed,
                                                seed_transpiler=self.seed)
        self.optimizer = COBYLA(maxiter=1)
        self.qmolecule1, self.core1, self.qubit_op1, self.var_form1, self.algo1\
            = self._create_components_for_tests(path='test_oovqe_h4.hdf5',
                                                freeze_core=False, two_qubit_reduction=False,
                                                initial_point=self.initial_point1)
        self.qmolecule2, self.core2, self.qubit_op2, self.var_form2, self.algo2\
            = self._create_components_for_tests(path='test_oovqe_lih.hdf5',
                                                freeze_core=True, two_qubit_reduction=True,
                                                initial_point=None)
        self.qmolecule3, self.core3, self.qubit_op3, self.var_form3, self.algo3\
            = self._create_components_for_tests(path='test_oovqe_h4_uhf.hdf5',
                                                freeze_core=False, two_qubit_reduction=False,
                                                initial_point=self.initial_point1)

    def _create_components_for_tests(self, path='', freeze_core=False,
                                     two_qubit_reduction=False, initial_point=None):
        """ Instantiate classes necessary to run the test out of HDF5 files of QMolecules."""

        driver = HDF5Driver(hdf5_input=self.get_resource_path(path))
        qmolecule = driver.run()
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.PARITY,
                           two_qubit_reduction=two_qubit_reduction,
                           freeze_core=freeze_core,
                           orbital_reduction=[])
        algo_input = core.run(qmolecule)
        qubit_op = algo_input[0]
        init_state = HartreeFock(
            num_orbitals=core._molecule_info['num_orbitals'],
            qubit_mapping=core._qubit_mapping,
            two_qubit_reduction=core._two_qubit_reduction,
            num_particles=core._molecule_info['num_particles'])
        var_form = UCCSD(
            num_orbitals=core._molecule_info['num_orbitals'],
            num_particles=core._molecule_info['num_particles'],
            active_occupied=None, active_unoccupied=None,
            initial_state=init_state,
            qubit_mapping=core._qubit_mapping,
            two_qubit_reduction=core._two_qubit_reduction,
            num_time_slices=1,
            method_doubles='pucc',
            same_spin_doubles=False,
            method_singles='both',
            skip_commute_test=True,
            excitation_type='d')
        algo = OOVQE(operator=qubit_op,
                     var_form=var_form,
                     optimizer=self.optimizer,
                     core=core,
                     qmolecule=qmolecule,
                     expectation=MatrixExpectation(),
                     initial_point=initial_point)

        return qmolecule, core, qubit_op, var_form, algo

    def test_orbital_rotations(self):
        """Test that orbital rotations are performed correctly."""

        self.algo1.optimizer.maxiter = 1
        algo_result = self.algo1.run(self.quantum_instance)
        self.assertAlmostEqual(algo_result['optimal_value'], self.energy1_rotation, 4)

    def test_oovqe(self):
        """Test the simultaneous optimization of orbitals and ansatz parameters with OOVQE using
        BasicAer's statevector_simulator."""

        self.algo1.optimizer.maxiter = 3
        self.algo1.optimizer.rhobeg = 0.01
        algo_result = self.algo1.run(self.quantum_instance)
        self.assertIsInstance(algo_result, OOVQEResult)
        self.assertIsNotNone(algo_result.optimal_point_orbitals)
        self.assertLessEqual(algo_result['optimal_value'], self.energy1)

    def test_iterative_oovqe(self):
        """Test the iterative OOVQE using BasicAer's statevector_simulator."""

        self.algo1.optimizer.maxiter = 2
        self.algo1.optimizer.rhobeg = 0.01
        self.algo1._iterative_oo = True
        self.algo1._iterative_oo_iterations = 2
        algo_result = self.algo1.run(self.quantum_instance)
        self.assertLessEqual(algo_result['optimal_value'], self.energy1)

    def test_oovqe_with_frozen_core(self):
        """Test the OOVQE with frozen core approximation."""

        self.algo2.optimizer.maxiter = 2
        self.algo2.optimizer.rhobeg = 1
        algo_result = self.algo2.run(self.quantum_instance)
        self.assertLessEqual(algo_result['optimal_value'] + self.core2._energy_shift +
                             self.core2._nuclear_repulsion_energy, self.energy2)

    def test_oovqe_with_unrestricted_hf(self):
        """Test the OOVQE with unrestricted HF method."""

        self.algo3.optimizer.maxiter = 2
        self.algo3.optimizer.rhobeg = 0.01
        algo_result = self.algo3.run(self.quantum_instance)
        self.assertLessEqual(algo_result['optimal_value'], self.energy3)


if __name__ == '__main__':
    unittest.main()
