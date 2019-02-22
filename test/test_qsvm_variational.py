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
import unittest

import numpy as np

from test.common import QiskitAquaTestCase
from qiskit.aqua import get_aer_backend
from qiskit.aqua.input import SVMInput
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.algorithms import QSVMVariational
from qiskit.aqua.components.optimizers import SPSA, COBYLA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.variational_forms import RYRZ, RY


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
        self.ref_prediction_a_probs = [[0.53710938, 0.46289062]]
        self.ref_prediction_a_label = [0]

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
        minibatching_ref_opt_params = np.asarray([-18.8965, -36.9197,   2.3568,  14.9087,  43.8633,   7.1394,
                                                  12.1298,  23.7219,  36.5693,  21.0561,  37.5775, -22.4538,
                                                  41.1386, -54.3177, -27.3063,  31.3858])
        minibatching_ref_train_loss = 1.02335933

        np.testing.assert_array_almost_equal(result['opt_params'], minibatching_ref_opt_params, decimal=4)
        np.testing.assert_array_almost_equal(result['training_loss'], minibatching_ref_train_loss, decimal=8)

        self.assertEqual(result['testing_accuracy'], .5)

    def test_qsvm_variational_directly(self):
        np.random.seed(self.random_seed)
        backend = get_aer_backend('qasm_simulator')

        num_qubits = 2
        optimizer = SPSA(max_trials=10, save_steps=1, c0=4.0, skip_calibration=True)
        feature_map = SecondOrderExpansion(num_qubits=num_qubits, depth=2)
        var_form = RYRZ(num_qubits=num_qubits, depth=3)

        svm = QSVMVariational(optimizer, feature_map, var_form, self.training_data, self.testing_data)
        svm.random_seed = self.random_seed
        quantum_instance = QuantumInstance(backend, shots=1024, seed=self.random_seed, seed_mapper=self.random_seed)
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
        np.testing.assert_array_almost_equal(predicted_probs, self.ref_prediction_a_probs, decimal=8)
        np.testing.assert_array_equal(predicted_labels, self.ref_prediction_a_label)

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

    def test_qsvm_variational_callback(self):

        tmp_filename = 'qsvm_callback_test.csv'
        is_file_exist = os.path.exists(self._get_resource_path(tmp_filename))
        if is_file_exist:
            os.remove(self._get_resource_path(tmp_filename))

        def store_intermediate_result(eval_count, parameters, cost, batch_index):
            with open(self._get_resource_path(tmp_filename), 'a') as f:
                content = "{},{},{:.5f},{}".format(eval_count, parameters, cost, batch_index)
                print(content, file=f, flush=True)

        np.random.seed(self.random_seed)
        backend = get_aer_backend('qasm_simulator')

        num_qubits = 2
        optimizer = COBYLA(maxiter=3)
        feature_map = SecondOrderExpansion(num_qubits=num_qubits, depth=2)
        var_form = RY(num_qubits=num_qubits, depth=1)

        svm = QSVMVariational(optimizer, feature_map, var_form, self.training_data,
                              self.testing_data, callback=store_intermediate_result)
        svm.random_seed = self.random_seed
        quantum_instance = QuantumInstance(backend, shots=1024, seed=self.random_seed, seed_mapper=self.random_seed)
        svm.run(quantum_instance)

        is_file_exist = os.path.exists(self._get_resource_path(tmp_filename))
        self.assertTrue(is_file_exist, "Does not store content successfully.")

        # check the content
        ref_content = [
            ["0", "[ 0.18863864 -1.08197582  1.74432295  1.29765602]", "0.53367", "0"],
            ["1", "[ 1.18863864 -1.08197582  1.74432295  1.29765602]", "0.57261", "1"],
            ["2", "[ 0.18863864 -0.08197582  1.74432295  1.29765602]", "0.47137", "2"]
            ]
        with open(self._get_resource_path(tmp_filename)) as f:
            idx = 0
            for record in f.readlines():
                eval_count, parameters, cost, batch_index = record.split(",")
                self.assertEqual(eval_count.strip(), ref_content[idx][0])
                self.assertEqual(parameters, ref_content[idx][1])
                self.assertEqual(cost.strip(), ref_content[idx][2])
                self.assertEqual(batch_index.strip(), ref_content[idx][3])
                idx += 1
        if is_file_exist:
            os.remove(self._get_resource_path(tmp_filename))


if __name__ == '__main__':
    unittest.main()
