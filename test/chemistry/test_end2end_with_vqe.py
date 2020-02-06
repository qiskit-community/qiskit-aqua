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

""" Test End to End with VQE """

import unittest
from test.chemistry import QiskitChemistryTestCase
from parameterized import parameterized
import qiskit
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms.adaptive import VQE
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.optimizers import COBYLA, SPSA
from qiskit.chemistry.drivers import HDF5Driver
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType


class TestEnd2End(QiskitChemistryTestCase):
    """End2End VQE tests."""

    def setUp(self):
        super().setUp()
        driver = HDF5Driver(hdf5_input=self.get_resource_path('test_driver_hdf5.hdf5'))
        self.qmolecule = driver.run()

        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.PARITY,
                           two_qubit_reduction=True,
                           freeze_core=False,
                           orbital_reduction=[])

        self.qubit_op, self.aux_ops = core.run(self.qmolecule)
        self.reference_energy = -1.857275027031588

    @parameterized.expand([
        ['COBYLA_M', 'COBYLA', qiskit.BasicAer.get_backend('statevector_simulator'), 1],
        ['COBYLA_P', 'COBYLA', qiskit.BasicAer.get_backend('statevector_simulator'), 1],
        # ['SPSA_P', 'SPSA', qiskit.BasicAer.get_backend('qasm_simulator'), 'paulis', 1024],
        # ['SPSA_GP', 'SPSA', qiskit.BasicAer.get_backend('qasm_simulator'), 'grouped_paulis', 1024]
    ])
    def test_end2end_h2(self, name, optimizer, backend, shots):
        """ end to end h2 """
        del name  # unused
        if optimizer == 'COBYLA':
            optimizer = COBYLA()
            optimizer.set_options(maxiter=1000)
        elif optimizer == 'SPSA':
            optimizer = SPSA(max_trials=2000)

        ryrz = RYRZ(self.qubit_op.num_qubits, depth=3, entanglement='full')
        vqe = VQE(self.qubit_op, ryrz, optimizer, aux_operators=self.aux_ops)
        quantum_instance = QuantumInstance(backend, shots=shots)
        results = vqe.run(quantum_instance)
        self.assertAlmostEqual(results['energy'], self.reference_energy, places=4)


if __name__ == '__main__':
    unittest.main()
