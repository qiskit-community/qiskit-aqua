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

""" Test of the Adaptive VQE implementation with the adaptive UCCSD variational form """

import unittest
from test.chemistry import QiskitChemistryTestCase

from qiskit.aqua import aqua_globals
from qiskit.aqua.components.optimizers import L_BFGS_B
from qiskit.aqua.operators.legacy.op_converter import to_weighted_pauli_operator
from qiskit.aqua.operators.legacy.weighted_pauli_operator import Z2Symmetries
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.algorithms import VQEAdapt
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import QiskitChemistryError


class TestVQEAdaptUCCSD(QiskitChemistryTestCase):
    """ Test Adaptive VQE with UCCSD"""
    def setUp(self):
        super().setUp()
        # np.random.seed(50)
        self.seed = 50
        aqua_globals.random_seed = self.seed
        try:
            driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                                 unit=UnitsType.ANGSTROM,
                                 basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')
            return

        molecule = driver.run()
        self.num_particles = molecule.num_alpha + molecule.num_beta
        self.num_spin_orbitals = molecule.num_orbitals * 2
        fer_op = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
        map_type = 'PARITY'
        qubit_op = fer_op.mapping(map_type)
        self.qubit_op = Z2Symmetries.two_qubit_reduction(to_weighted_pauli_operator(qubit_op),
                                                         self.num_particles)
        self.num_qubits = self.qubit_op.num_qubits
        self.init_state = HartreeFock(self.num_qubits, self.num_spin_orbitals, self.num_particles)
        self.var_form_base = None

    def test_uccsd_adapt(self):
        """ UCCSD test for adaptive features """
        self.var_form_base = UCCSD(self.num_qubits, 1, self.num_spin_orbitals,
                                   self.num_particles, initial_state=self.init_state)
        self.var_form_base.manage_hopping_operators()
        # assert that the excitation pool exists
        self.assertIsNotNone(self.var_form_base.excitation_pool)
        # assert that the hopping ops list has been reset to be empty
        self.assertEqual(self.var_form_base._hopping_ops, [])

    def test_vqe_adapt(self):
        """ VQEAdapt test """
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

        self.var_form_base = UCCSD(self.num_qubits, 1, self.num_spin_orbitals,
                                   self.num_particles, initial_state=self.init_state)
        backend = Aer.get_backend('statevector_simulator')
        optimizer = L_BFGS_B()
        algorithm = VQEAdapt(self.qubit_op, self.var_form_base, optimizer,
                             threshold=0.00001, delta=0.1)
        result = algorithm.run(backend)
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=2)
        self.assertIsNotNone(result.num_iterations)
        self.assertIsNotNone(result.final_max_gradient)
        self.assertIsNotNone(result.finishing_criterion)


if __name__ == '__main__':
    unittest.main()
