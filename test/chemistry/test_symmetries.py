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

"""
Test of Symmetry UCCSD processing.
"""

import itertools
from test.chemistry.common import QiskitChemistryTestCase
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, Operator
from qiskit.aqua.algorithms.adaptive import VQE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock

# from qiskit.chemistry import set_qiskit_chemistry_logging
# import logging


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
        self.symmetries, self.sq_paulis, self.cliffords, self.sq_list = self.qubit_op.find_Z2_symmetries()

        self.reference_energy = -7.882096489442

    def test_symmetries(self):
        labels = [symm.to_label() for symm in self.symmetries]
        self.assertSequenceEqual(labels, ['ZIZIZIZI', 'ZZIIZZII'])

    def test_sq_paulis(self):
        labels = [sq.to_label() for sq in self.sq_paulis]
        self.assertSequenceEqual(labels, ['IIIIIIXI', 'IIIIIXII'])

    def test_cliffords(self):
        self.assertEqual(2, len(self.cliffords))

    def test_sq_list(self):
        self.assertSequenceEqual(self.sq_list, [1, 2])

    def test_tapered_op(self):
        # set_qiskit_chemistry_logging(logging.DEBUG)
        tapered_ops = []
        for coeff in itertools.product([1, -1], repeat=len(self.sq_list)):
            tapered_op = Operator.qubit_tapering(self.qubit_op, self.cliffords, self.sq_list, list(coeff))
            tapered_ops.append((list(coeff), tapered_op))

        smallest_idx = 0  # Prior knowledge of which tapered_op has ground state
        the_tapered_op = tapered_ops[smallest_idx][1]
        the_coeff = tapered_ops[smallest_idx][0]

        optimizer = SLSQP(maxiter=1000)

        init_state = HartreeFock(num_qubits=the_tapered_op.num_qubits,
                                 num_orbitals=self.core._molecule_info['num_orbitals'],
                                 qubit_mapping=self.core._qubit_mapping,
                                 two_qubit_reduction=self.core._two_qubit_reduction,
                                 num_particles=self.core._molecule_info['num_particles'],
                                 sq_list=self.sq_list)

        var_form = UCCSD(num_qubits=the_tapered_op.num_qubits, depth=1,
                         num_orbitals=self.core._molecule_info['num_orbitals'],
                         num_particles=self.core._molecule_info['num_particles'],
                         active_occupied=None, active_unoccupied=None,
                         initial_state=init_state,
                         qubit_mapping=self.core._qubit_mapping,
                         two_qubit_reduction=self.core._two_qubit_reduction,
                         num_time_slices=1,
                         cliffords=self.cliffords, sq_list=self.sq_list,
                         tapering_values=the_coeff, symmetries=self.symmetries)

        algo = VQE(the_tapered_op, var_form, optimizer, 'matrix')

        backend = BasicAer.get_backend('statevector_simulator')
        quantum_instance = QuantumInstance(backend=backend)

        algo_result = algo.run(quantum_instance)

        lines, result = self.core.process_algorithm_result(algo_result)

        self.assertAlmostEqual(result['energy'], self.reference_energy, places=6)
