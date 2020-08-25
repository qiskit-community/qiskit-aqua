# -*- coding: utf-8 -*-

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

""" Test OOVQE """

import unittest
from test.aqua import QiskitAquaTestCase
from ddt import ddt
from qiskit.aqua import aqua_globals
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.chemistry.core import Hamiltonian
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.core import TransformationType, QubitMappingType
from qiskit.chemistry.algorithms.minimum_eigen_solvers import OOVQE
from qiskit.aqua.operators.expectations import MatrixExpectation
from qiskit import Aer

@ddt
class TestOOVQE(QiskitAquaTestCase):
    """ Test of the OOVQE algorithm"""

    def setUp(self):
        super().setUp()
        self.energy_vqe = -2.77  # energy of the VQE with pUCCD ansatz and LBFGSB optimizer
        self.initial_point = [0.039374, -0.47225463, -0.61891996, 0.02598386,   0.79045546,
                              -0.04134567,  0.04944946, -0.02971617, -0.00374005, 0.77542149]
        self.seed = 50
        aqua_globals.random_seed = self.seed
        driver = PySCFDriver(
            atom="H 1.738000 .0 .0; H .15148 1.73139 .0; H -1.738 .0 .0; H -0.15148 -1.73139 .0",
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=0,
            basis='sto3g')
        self.qmolecule = driver.run()
        self.core = Hamiltonian(transformation=TransformationType.FULL,
                                qubit_mapping=QubitMappingType.PARITY,
                                two_qubit_reduction=False,
                                freeze_core=False,
                                orbital_reduction=[])

        algo_input = self.core.run(self.qmolecule)
        self.qubit_op = algo_input[0]

        init_state = HartreeFock(
            num_orbitals=self.core._molecule_info['num_orbitals'],
            qubit_mapping=self.core._qubit_mapping,
            two_qubit_reduction=self.core._two_qubit_reduction,
            num_particles=self.core._molecule_info['num_particles'])

        self.var_form = UCCSD(
            num_orbitals=self.core._molecule_info['num_orbitals'],
            num_particles=self.core._molecule_info['num_particles'],
            active_occupied=None, active_unoccupied=None,
            initial_state=init_state,
            qubit_mapping=self.core._qubit_mapping,
            two_qubit_reduction=self.core._two_qubit_reduction,
            num_time_slices=1,
            method_doubles='pucc',
            same_spin_doubles=False,
            method_singles='both',
            skip_commute_test=True,
            excitation_type='d')

        self.quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'),
                                           shots=1,
                                           seed_simulator=self.seed,
                                           seed_transpiler=self.seed)
        self.optimizer = COBYLA(maxiter=1)
        self.algo = OOVQE(operator=self.qubit_op,
                     var_form=self.var_form,
                     optimizer=self.optimizer,
                     core=self.core,
                     qmolecule=self.qmolecule,
                     expectation=MatrixExpectation(),
                     initial_point=self.initial_point
                     )

    def test_orbital_rotations(self):
        """Test that orbital rotations are performed correctly."""

        self.algo.optimizer.maxiter = 1
        algo_result = self.algo.run(self.quantum_instance)
        self.assertAlmostEqual(algo_result['optimal_value'], -3.0104, 4)

    def test_oovqe(self):
        """Test the simultaneous optimization of orbitals and ansatz parameters with OOVQE using
        Aer's statevector_simulator."""

        self.algo.optimizer.maxiter = 3
        self.algo.optimizer.rhobeg = 0.01
        algo_result = self.algo.run(self.quantum_instance)
        self.assertLessEqual(algo_result['optimal_value'], self.energy_vqe)

    def test_iterative_oovqe(self):
        """Test the iterative OOVQE using Aer's statevector_simulator."""

        self.algo.optimizer.maxiter = 2
        self.algo.optimizer.rhobeg = 0.01
        self.algo.iterative_oo = True
        self.algo.iterative_oo_iterations = 2
        algo_result = self.algo.run(self.quantum_instance)
        self.assertLessEqual(algo_result['optimal_value'], self.energy_vqe)

if __name__ == '__main__':
    unittest.main()
