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

import numpy as np
from parameterized import parameterized
from qiskit_aqua import get_aer_backend

from test.common import QiskitAquaTestCase
from qiskit_aqua import Operator, run_algorithm, QuantumInstance
from qiskit_aqua.input import EnergyInput
from qiskit_aqua.components.variational_forms import RY
from qiskit_aqua.components.optimizers import L_BFGS_B
from qiskit_aqua.components.initial_states import Zero
from qiskit_aqua.algorithms import VQE


class TestVQE(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()
        np.random.seed(50)
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        qubit_op = Operator.load_from_dict(pauli_dict)
        self.algo_input = EnergyInput(qubit_op)

    def test_vqe_via_run_algorithm(self):
        params = {
            'algorithm': {'name': 'VQE'},
            'backend': {'name': 'statevector_simulator', 'shots': 1},
        }
        result = run_algorithm(params, self.algo_input)
        self.assertAlmostEqual(result['energy'], -1.85727503)
        np.testing.assert_array_almost_equal(result['eigvals'], [-1.85727503], 5)
        np.testing.assert_array_almost_equal(result['opt_params'],
                                             [-0.58294401, -1.86141794, -1.97209632, -0.54796022,
                                              -0.46945572, 2.60114794, -1.15637845,  1.40498879,
                                              1.14479635, -0.48416694, -0.66608349, -1.1367579,
                                              -2.67097002, 3.10214631, 3.10000313, 0.37235089], 5)
        self.assertIn('eval_count', result)
        self.assertIn('eval_time', result)

    @parameterized.expand([
        ['CG', 5, True],
        ['CG', 5, False],
        ['COBYLA', 5, False],
        ['L_BFGS_B', 5, True],
        ['L_BFGS_B', 5, False],
        ['NELDER_MEAD', 5, False],
        ['POWELL', 5, False],
        ['SLSQP', 5, True],
        ['SLSQP', 5, False],
        ['SPSA', 3, True],
        ['SPSA', 3, False],
        ['TNC', 2, True],
        ['TNC', 2, False]
    ])
    def test_vqe_optimizers(self, name, places, batch_mode):
        params = {
            'algorithm': {'name': 'VQE', 'batch_mode': batch_mode},
            'optimizer': {'name': name},
            'backend': {'name': 'statevector_simulator', 'shots': 1}
        }
        result = run_algorithm(params, self.algo_input)
        self.assertAlmostEqual(result['energy'], -1.85727503, places=places)

    @parameterized.expand([
        ['RY', 5],
        ['RYRZ', 5]
    ])
    def test_vqe_var_forms(self, name, places):
        backend = get_aer_backend('statevector_simulator')
        params = {
            'algorithm': {'name': 'VQE'},
            'variational_form': {'name': name},
            'backend': {'shots': 1}
        }
        result = run_algorithm(params, self.algo_input, backend=backend)
        self.assertAlmostEqual(result['energy'], -1.85727503, places=places)


    @parameterized.expand([
        [True],
        [False]
    ])
    def test_vqe_direct(self, batch_mode):
        backend = get_aer_backend('statevector_simulator')
        num_qubits = self.algo_input.qubit_op.num_qubits
        init_state = Zero(num_qubits)
        var_form = RY(num_qubits, 3, initial_state=init_state)
        optimizer = L_BFGS_B()
        algo = VQE(self.algo_input.qubit_op, var_form, optimizer, 'matrix', batch_mode=batch_mode)
        quantum_instance = QuantumInstance(backend)
        result = algo.run(quantum_instance)
        self.assertAlmostEqual(result['energy'], -1.85727503)


if __name__ == '__main__':
    unittest.main()
