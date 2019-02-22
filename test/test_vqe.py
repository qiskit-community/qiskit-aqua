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
import os

import numpy as np
from parameterized import parameterized

from test.common import QiskitAquaTestCase
from qiskit.aqua import get_aer_backend
from qiskit.aqua import Operator, run_algorithm, QuantumInstance
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua.components.optimizers import L_BFGS_B, COBYLA
from qiskit.aqua.components.initial_states import Zero
from qiskit.aqua.algorithms import VQE


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

        coupling_map = [[0, 1]]
        basis_gates = ['u1', 'u2', 'u3', 'cx', 'id']

        params = {
            'algorithm': {'name': 'VQE'},
            'backend': {'name': 'statevector_simulator',
                        'provider': 'qiskit.Aer',
                        'shots': 1,
                        'coupling_map': coupling_map,
                        'basis_gates': basis_gates},
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
        backend = get_aer_backend('statevector_simulator')
        params = {
            'algorithm': {'name': 'VQE', 'batch_mode': batch_mode},
            'optimizer': {'name': name},
            'backend': {'shots': 1}
        }
        result = run_algorithm(params, self.algo_input, backend=backend)
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
        algo = VQE(self.algo_input.qubit_op, var_form, optimizer, 'paulis', batch_mode=batch_mode)
        quantum_instance = QuantumInstance(backend)
        result = algo.run(quantum_instance)
        self.assertAlmostEqual(result['energy'], -1.85727503)

    def test_vqe_callback(self):

        tmp_filename = 'vqe_callback_test.csv'
        is_file_exist = os.path.exists(self._get_resource_path(tmp_filename))
        if is_file_exist:
            os.remove(self._get_resource_path(tmp_filename))

        def store_intermediate_result(eval_count, parameters, mean, std):
            with open(self._get_resource_path(tmp_filename), 'a') as f:
                content = "{},{},{:.5f},{:.5f}".format(eval_count, parameters, mean, std)
                print(content, file=f, flush=True)

        backend = get_aer_backend('qasm_simulator')
        num_qubits = self.algo_input.qubit_op.num_qubits
        init_state = Zero(num_qubits)
        var_form = RY(num_qubits, 1, initial_state=init_state)
        optimizer = COBYLA(maxiter=3)
        algo = VQE(self.algo_input.qubit_op, var_form, optimizer, 'paulis',
                   callback=store_intermediate_result)
        algo.random_seed = 50
        quantum_instance = QuantumInstance(backend, seed_mapper=50, shots=1024, seed=50)
        algo.run(quantum_instance)

        is_file_exist = os.path.exists(self._get_resource_path(tmp_filename))
        self.assertTrue(is_file_exist, "Does not store content successfully.")

        # check the content
        ref_content = [["1", "[-0.03391886 -1.70850424 -1.53640265 -0.65137839]", "-0.59622", "0.01546"],
                       ["2", "[ 0.96608114 -1.70850424 -1.53640265 -0.65137839]", "-0.77452", "0.01692"],
                       ["3", "[ 0.96608114 -0.70850424 -1.53640265 -0.65137839]", "-0.80327", "0.01519"]
                       ]
        with open(self._get_resource_path(tmp_filename)) as f:
            idx = 0
            for record in f.readlines():
                eval_count, parameters, mean, std = record.split(",")
                self.assertEqual(eval_count.strip(), ref_content[idx][0])
                self.assertEqual(parameters, ref_content[idx][1])
                self.assertEqual(mean.strip(), ref_content[idx][2])
                self.assertEqual(std.strip(), ref_content[idx][3])
                idx += 1
        if is_file_exist:
            os.remove(self._get_resource_path(tmp_filename))


if __name__ == '__main__':
    unittest.main()
