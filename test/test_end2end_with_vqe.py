# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import unittest
from parameterized import parameterized
import qiskit
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms.adaptive import VQE
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.optimizers import COBYLA, SPSA
from test.common import QiskitChemistryTestCase
from qiskit.chemistry.drivers import HDF5Driver
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType


class TestEnd2End(QiskitChemistryTestCase):
    """End2End tests."""

    def setUp(self):
        driver = HDF5Driver(hdf5_input=self._get_resource_path('test_driver_hdf5.hdf5'))
        self.qmolecule = driver.run()

        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.PARITY,
                           two_qubit_reduction=True,
                           freeze_core=False,
                           orbital_reduction=[],
                           max_workers=4)

        self.algo_input = core.run(self.qmolecule)
        self.reference_energy = -1.857275027031588

    @parameterized.expand([
        ['COBYLA_M', 'COBYLA', qiskit.Aer.get_backend('statevector_simulator'), 'matrix', 1],
        ['COBYLA_P', 'COBYLA', qiskit.Aer.get_backend('statevector_simulator'), 'paulis', 1],
        # ['SPSA_P', 'SPSA', 'qasm_simulator', 'paulis', 1024],
        # ['SPSA_GP', 'SPSA', 'qasm_simulator', 'grouped_paulis', 1024]
    ])
    def test_end2end_h2(self, name, optimizer, backend, mode, shots):

        if optimizer == 'COBYLA':
            optimizer = COBYLA()
            optimizer.set_options(maxiter=1000)
        elif optimizer == 'SPSA':
            optimizer = SPSA(max_trials=2000)

        ryrz = RYRZ(self.algo_input.qubit_op.num_qubits, depth=3, entanglement='full')
        vqe = VQE(self.algo_input.qubit_op, ryrz, optimizer, mode, aux_operators=self.algo_input.aux_ops)
        quantum_instance = QuantumInstance(backend, shots=shots)
        results = vqe.run(quantum_instance)
        self.assertAlmostEqual(results['energy'], self.reference_energy, places=4)


if __name__ == '__main__':
    unittest.main()
