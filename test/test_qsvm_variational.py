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

import os

import numpy as np
from qiskit.aqua import get_aer_backend
from qiskit.qobj import RunConfig
from test.common import QiskitAquaTestCase
from qiskit.aqua.input import SVMInput
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.algorithms import QSVMVariational
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.variational_forms import RYRZ


class TestQSVMVariational(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()
        self.random_seed = 10598
        self.training_data = {'A': np.asarray([[2.95309709, 2.51327412], [3.14159265, 4.08407045]]),
                              'B': np.asarray([[4.08407045, 2.26194671], [4.46106157, 2.38761042]])}
        self.testing_data = {'A': np.asarray([[3.83274304, 2.45044227]]),
                             'B': np.asarray([[3.89557489, 0.31415927]])}

        self.ref_opt_params = np.array([-0.09936191, -1.26202073,  1.30316646,  3.24053034, -0.50731743,
                                        -0.6853292,  2.57404557,  1.74873317,  1.62238446, -1.83326183,
                                        4.48499251,  0.21433137, -1.76288916, -0.15767913,  1.86321388,
                                        0.27216782])
        self.ref_train_loss = 1.4088445273265953

        self.svm_input = SVMInput(self.training_data, self.testing_data)

    def test_qsvm_variational_via_run_algorithm(self):
        np.random.seed(self.random_seed)
        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'algorithm': {'name': 'QSVM.Variational'},
            'backend': {'name': 'qasm_simulator', 'shots': 1024},
            'optimizer': {'name': 'SPSA', 'max_trials': 10, 'save_steps': 1},
            'variational_form': {'name': 'RYRZ', 'depth': 3},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2}
        }
        result = run_algorithm(params, self.svm_input)

        np.testing.assert_array_almost_equal(result['opt_params'], self.ref_opt_params, decimal=4)
        np.testing.assert_array_almost_equal(result['training_loss'], self.ref_train_loss, decimal=8)

        self.assertEqual(result['testing_accuracy'], 1.0)

    def test_qsvm_variational_with_minibatching(self):
        np.random.seed(self.random_seed)
        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'algorithm': {'name': 'QSVM.Variational', 'minibatch_size': 2},
            'backend': {'name': 'qasm_simulator', 'shots': 1024},
            'optimizer': {'name': 'SPSA', 'max_trials': 30, 'save_steps': 1},
            'variational_form': {'name': 'RYRZ', 'depth': 3},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2}
        }
        result = run_algorithm(params, self.svm_input)

        # The results will differ from the above even though the batch size is larger than the trainingset size due
        # to the shuffle during minibatching
        minibatching_ref_opt_params = np.asarray([3.3604,   1.9649,   7.8314,  -4.738,  -3.1619,  15.6005,  -4.918,
                                                  10.811,   9.3323,   7.1805,  13.5328,   9.3692,  -7.2276,
                                                  -2.0618,  -0.2071, -11.6123])
        minibatching_ref_train_loss = 0.66402647

        np.testing.assert_array_almost_equal(result['opt_params'], minibatching_ref_opt_params, decimal=4)
        np.testing.assert_array_almost_equal(result['training_loss'], minibatching_ref_train_loss, decimal=8)

        self.assertEqual(result['testing_accuracy'], 1.0)

    def test_qsvm_variational_directly(self):
        np.random.seed(self.random_seed)
        backend = get_aer_backend('qasm_simulator')

        num_qubits = 2
        optimizer = SPSA(max_trials=10, save_steps=1, c0=4.0, skip_calibration=True)
        feature_map = SecondOrderExpansion(num_qubits=num_qubits, depth=2)
        var_form = RYRZ(num_qubits=num_qubits, depth=3)

        svm = QSVMVariational(optimizer, feature_map, var_form, self.training_data, self.testing_data)
        svm.random_seed = self.random_seed
        run_config = RunConfig(shots=1024, max_credits=10, memory=False, seed=self.random_seed)
        quantum_instance = QuantumInstance(backend, run_config, seed_mapper=self.random_seed)
        result = svm.run(quantum_instance)

        np.testing.assert_array_almost_equal(result['opt_params'], self.ref_opt_params, decimal=4)
        np.testing.assert_array_almost_equal(result['training_loss'], self.ref_train_loss, decimal=8)

        self.assertEqual(result['testing_accuracy'], 1.0)

        file_path = self._get_resource_path('qsvm_variational_test.npz')
        svm.save_model(file_path)

        self.assertTrue(os.path.exists(file_path))

        loaded_svm = QSVMVariational(optimizer, feature_map, var_form, self.training_data, None)
        loaded_svm.load_model(file_path)

        np.testing.assert_array_almost_equal(
            loaded_svm.ret['opt_params'], self.ref_opt_params, decimal=4)

        loaded_test_acc = loaded_svm.test(svm.test_dataset[0], svm.test_dataset[1], quantum_instance)
        self.assertEqual(result['testing_accuracy'], loaded_test_acc)

        predicted_probs, predicted_labels = loaded_svm.predict(self.testing_data['A'], quantum_instance)
        print(predicted_probs)
        print(predicted_labels)
        np.testing.assert_array_almost_equal(prediction, [0, 1], decimal=8)

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
