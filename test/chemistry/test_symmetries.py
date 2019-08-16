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

""" Test of Symmetry UCCSD processing """

from test.chemistry.common import QiskitChemistryTestCase
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import Z2Symmetries
from qiskit.aqua.algorithms.adaptive import VQE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock


class TestSymmetries(QiskitChemistryTestCase):
    """Test for symmetry processing."""

    def setUp(self):
        super().setUp()
        try:
            driver = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 1.6',
                                 unit=UnitsType.ANGSTROM,
                                 charge=0,
                                 spin=0,
                                 basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')
        self.qmolecule = driver.run()
        self.core = Hamiltonian(transformation=TransformationType.FULL,
                                qubit_mapping=QubitMappingType.PARITY,
                                two_qubit_reduction=True,
                                freeze_core=True,
                                orbital_reduction=[])
        self.qubit_op, _ = self.core.run(self.qmolecule)
        self.z2_symmetries = Z2Symmetries.find_Z2_symmetries(self.qubit_op)

        self.reference_energy = -7.882096489442

    def test_symmetries(self):
        """ symmetries test """
        labels = [symm.to_label() for symm in self.z2_symmetries.symmetries]
        self.assertSequenceEqual(labels, ['ZIZIZIZI', 'ZZIIZZII'])

    def test_sq_paulis(self):
        """ sq paulis test """
        labels = [sq.to_label() for sq in self.z2_symmetries.sq_paulis]
        self.assertSequenceEqual(labels, ['IIIIIIXI', 'IIIIIXII'])

    def test_cliffords(self):
        """ clifford test """
        self.assertEqual(2, len(self.z2_symmetries.cliffords))

    def test_sq_list(self):
        """ sq list test """
        self.assertSequenceEqual(self.z2_symmetries.sq_list, [1, 2])

    def test_tapered_op(self):
        """ tapered op test """
        tapered_ops = self.z2_symmetries.taper(self.qubit_op)
        smallest_idx = 0  # Prior knowledge of which tapered_op has ground state
        the_tapered_op = tapered_ops[smallest_idx]

        optimizer = SLSQP(maxiter=1000)

        init_state = HartreeFock(num_qubits=the_tapered_op.num_qubits,
                                 num_orbitals=self.core._molecule_info['num_orbitals'],
                                 qubit_mapping=self.core._qubit_mapping,
                                 two_qubit_reduction=self.core._two_qubit_reduction,
                                 num_particles=self.core._molecule_info['num_particles'],
                                 sq_list=the_tapered_op.z2_symmetries.sq_list)

        var_form = UCCSD(num_qubits=the_tapered_op.num_qubits, depth=1,
                         num_orbitals=self.core._molecule_info['num_orbitals'],
                         num_particles=self.core._molecule_info['num_particles'],
                         active_occupied=None, active_unoccupied=None,
                         initial_state=init_state,
                         qubit_mapping=self.core._qubit_mapping,
                         two_qubit_reduction=self.core._two_qubit_reduction,
                         num_time_slices=1,
                         z2_symmetries=the_tapered_op.z2_symmetries)

        algo = VQE(the_tapered_op, var_form, optimizer)

        backend = BasicAer.get_backend('statevector_simulator')
        quantum_instance = QuantumInstance(backend=backend)

        algo_result = algo.run(quantum_instance)

        _, result = self.core.process_algorithm_result(algo_result)

        self.assertAlmostEqual(result['energy'], self.reference_energy, places=6)
