# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of UCCSD and HartreeFock Aqua extensions """

from test.chemistry.common import QiskitChemistryTestCase
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.core import Hamiltonian, QubitMappingType, TransformationType

# pylint: disable=invalid-name


class TestUCCSDHartreeFock(QiskitChemistryTestCase):
    """Test for these aqua extensions."""

    def setUp(self):
        super().setUp()
        self.molecule = "H 0.000000 0.000000 0.735000;H 0.000000 0.000000 0.000000"
        self.driver = PySCFDriver(atom=self.molecule,
                                  unit=UnitsType.ANGSTROM,
                                  charge=0,
                                  spin=0,
                                  basis='631g')

        self.qmolecule = self.driver.run()

        self.core = Hamiltonian(transformation=TransformationType.FULL,
                                qubit_mapping=QubitMappingType.PARITY,
                                two_qubit_reduction=True,
                                freeze_core=True,
                                orbital_reduction=[])
        self.qubit_op, _ = self.core.run(self.qmolecule)

        self.reference_energy_pUCCD = -1.1434447924298028
        self.reference_energy_UCCD0 = -1.1476045878481704
        self.reference_energy_UCCD0full = -1.1515491334334347

        pass

    def test_uccsd_hf_qpUCCD(self):
        """ paired uccd test """
        self.qubit_op, _ = self.core.run(self.qmolecule)

        optimizer = SLSQP(maxiter=100)
        initial_state = HartreeFock(self.qubit_op.num_qubits,
                                    self.core.molecule_info['num_orbitals'],
                                    self.core.molecule_info['num_particles'],
                                    qubit_mapping=self.core._qubit_mapping,
                                    two_qubit_reduction=self.core._two_qubit_reduction)

        var_form = UCCSD(num_qubits=self.qubit_op.num_qubits, depth=1,
                         num_orbitals=self.core._molecule_info['num_orbitals'],
                         num_particles=self.core._molecule_info['num_particles'],
                         active_occupied=None, active_unoccupied=None,
                         initial_state=initial_state,
                         qubit_mapping=self.core._qubit_mapping,
                         two_qubit_reduction=self.core._two_qubit_reduction,
                         num_time_slices=1,
                         shallow_circuit_concat=False,
                         method_doubles='pucc',
                         exc_type='d'
                         )

        algo = VQE(self.qubit_op, var_form, optimizer)
        result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
        _, result = self.core.process_algorithm_result(result)
        self.assertAlmostEqual(result['energy'], self.reference_energy_pUCCD, places=6)

    def test_uccsd_hf_qUCCD0(self):
        """ singlet uccd test """
        self.qubit_op, _ = self.core.run(self.qmolecule)

        optimizer = SLSQP(maxiter=100)
        initial_state = HartreeFock(self.qubit_op.num_qubits,
                                    self.core.molecule_info['num_orbitals'],
                                    self.core.molecule_info['num_particles'],
                                    qubit_mapping=self.core._qubit_mapping,
                                    two_qubit_reduction=self.core._two_qubit_reduction)

        var_form = UCCSD(num_qubits=self.qubit_op.num_qubits, depth=1,
                         num_orbitals=self.core._molecule_info['num_orbitals'],
                         num_particles=self.core._molecule_info['num_particles'],
                         active_occupied=None, active_unoccupied=None,
                         initial_state=initial_state,
                         qubit_mapping=self.core._qubit_mapping,
                         two_qubit_reduction=self.core._two_qubit_reduction,
                         num_time_slices=1,
                         shallow_circuit_concat=False,
                         method_doubles='succ',
                         exc_type='d'
                         )

        algo = VQE(self.qubit_op, var_form, optimizer)
        result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
        _, result = self.core.process_algorithm_result(result)
        self.assertAlmostEqual(result['energy'], self.reference_energy_UCCD0, places=6)

    def test_uccsd_hf_qUCCD0full(self):
        """ singlet full uccd test """
        self.qubit_op, _ = self.core.run(self.qmolecule)

        optimizer = SLSQP(maxiter=100)
        initial_state = HartreeFock(self.qubit_op.num_qubits,
                                    self.core.molecule_info['num_orbitals'],
                                    self.core.molecule_info['num_particles'],
                                    qubit_mapping=self.core._qubit_mapping,
                                    two_qubit_reduction=self.core._two_qubit_reduction)

        var_form = UCCSD(num_qubits=self.qubit_op.num_qubits, depth=1,
                         num_orbitals=self.core._molecule_info['num_orbitals'],
                         num_particles=self.core._molecule_info['num_particles'],
                         active_occupied=None, active_unoccupied=None,
                         initial_state=initial_state,
                         qubit_mapping=self.core._qubit_mapping,
                         two_qubit_reduction=self.core._two_qubit_reduction,
                         num_time_slices=1,
                         shallow_circuit_concat=False,
                         method_doubles='succ_full',
                         exc_type='d'
                         )

        algo = VQE(self.qubit_op, var_form, optimizer)
        result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
        _, result = self.core.process_algorithm_result(result)
        self.assertAlmostEqual(result['energy'], self.reference_energy_UCCD0full, places=6)
