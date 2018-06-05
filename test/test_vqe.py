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
from collections import OrderedDict

import numpy as np
from qiskit.tools.qi.pauli import Pauli, label_to_pauli

from test.common import QISKitAcquaTestCase
from qiskit_acqua import Operator
from qiskit_acqua import get_variational_form_instance, get_algorithm_instance, get_optimizer_instance


class TestVQE(QISKitAcquaTestCase):
    """Operator tests."""

    def setUp(self):

        # create a H2 molecular:
        paulis = [[-1.052373245772859+0j, label_to_pauli('II')],
                    [0.39793742484318045+0j, label_to_pauli('ZI')],
                    [-0.39793742484318045+0j, label_to_pauli('IZ')],
                    [0.18093119978423156+0j, label_to_pauli('XX')],
                    [-0.01128010425623538+0j, label_to_pauli('ZZ')]
            ]
        self.qubit_op = Operator(paulis=paulis)
        self.gt_eigval = -1.857275027031588

    @parameterized.expand([
        ['COBYLA_M', 'COBYLA', 'local_statevector_simulator', 'matrix', 1],
        ['COBYLA_P', 'COBYLA', 'local_statevector_simulator', 'paulis', 1],
        ['SPSA_P', 'SPSA', 'local_qasm_simulator', 'paulis', 1024],
        ['SPSA_GP', 'SPSA', 'local_qasm_simulator', 'grouped_paulis', 1024]
    ])
    def test_minimization(self, name, optimizer, backend, mode, shots):

        opt = get_optimizer_instance(optimizer)
        if optimizer == 'COBYLA':
            opt.set_options(maxiter=1000)
        elif optimizer == 'SPSA':
            opt.init_args(max_trials=1000)

            opt.set_options(save_steps=25)
        var_form = get_variational_form_instance('RY')
        var_form.init_args(self.qubit_op.num_qubits, 3, entanglement = 'full')
        vqe = get_algorithm_instance('VQE')
        vqe.setup_quantum_backend(backend=backend, shots=shots)
        vqe.init_args(self.qubit_op, mode, var_form, opt)
        results = vqe.run()

        eigval = results['eigvals'][0]
        precision_place = 2 if shots > 1 else 7
        self.assertAlmostEqual(eigval, self.gt_eigval, places=precision_place)


if __name__ == '__main__':
    unittest.main()
