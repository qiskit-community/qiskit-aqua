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
from qiskit import BasicAer
from qiskit.aqua.input import SVMInput
from qiskit.aqua import run_algorithm, QuantumInstance, aqua_globals
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

        self.ref_opt_params = np.array([8.84487704, -4.75068608, -3.09321599, 6.15074807,
                                        -8.13322889, -10.03379214, 5.4842633, -0.80973346,
                                        -1.57635832, -9.36628893, -5.97527339, -2.65074375,
                                        -4.45536502, 10.86323401, 11.39789674, 3.65879025])
        self.ref_train_loss = 0.35346867
        self.ref_prediction_a_probs = [[0.55273438, 0.44726562]]
        self.ref_prediction_a_label = [0]

        self.svm_input = SVMInput(self.training_data, self.testing_data)

    def test_qsvm_variational_via_run_algorithm(self):
        np.random.seed(self.random_seed)
        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'algorithm': {'name': 'QSVM.Variational'},
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'qasm_simulator', 'shots': 1024},
            'optimizer': {'name': 'SPSA', 'max_trials': 10, 'save_steps': 1},
            'variational_form': {'name': 'RYRZ', 'depth': 3},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2}
        }
        result = run_algorithm(params, self.svm_input)

        np.testing.assert_array_almost_equal(result['opt_params'], self.ref_opt_params, decimal=8)
        np.testing.assert_array_almost_equal(result['training_loss'], self.ref_train_loss, decimal=8)

        self.assertEqual(result['testing_accuracy'], 0.5)

    def test_qsvm_variational_statevector_via_run_algorithm(self):
        np.random.seed(self.random_seed)
        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'algorithm': {'name': 'QSVM.Variational'},
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'statevector_simulator'},
            'optimizer': {'name': 'COBYLA'},
            'variational_form': {'name': 'RYRZ', 'depth': 3},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2}
        }
        result = run_algorithm(params, self.svm_input)
        ref_train_loss = 0.1059404
        np.testing.assert_array_almost_equal(result['training_loss'], ref_train_loss, decimal=4)

        self.assertEqual(result['testing_accuracy'], 0.5)

    def test_qsvm_variational_with_minibatching(self):
        np.random.seed(self.random_seed)
        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'algorithm': {'name': 'QSVM.Variational', 'minibatch_size': 2},
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'qasm_simulator', 'shots': 1024},
            'optimizer': {'name': 'SPSA', 'max_trials': 30, 'save_steps': 1},
            'variational_form': {'name': 'RYRZ', 'depth': 3},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2}
        }
        result = run_algorithm(params, self.svm_input)

        # The results will differ from the above even though the batch size is larger than the trainingset size due
        # to the shuffle during minibatching
        minibatching_ref_opt_params = np.array([2.4271, -21.5146, 4.4769, 21.0945, 8.4016, -9.7612,
                                                13.9343, 42.8698, 16.8601, -22.8767, 19.5411, -27.62,
                                                3.423, -25.9107, 21.0475, 21.895])

        minibatching_ref_train_loss = 0.6519675

        np.testing.assert_array_almost_equal(result['opt_params'], minibatching_ref_opt_params, decimal=4)
        np.testing.assert_array_almost_equal(result['training_loss'], minibatching_ref_train_loss, decimal=8)

        self.assertEqual(result['testing_accuracy'], 0.0)

    def test_qsvm_variational_directly(self):
        np.random.seed(self.random_seed)
        backend = BasicAer.get_backend('qasm_simulator')

        num_qubits = 2
        optimizer = SPSA(max_trials=10, save_steps=1, c0=4.0, skip_calibration=True)
        feature_map = SecondOrderExpansion(num_qubits=num_qubits, depth=2)
        var_form = RYRZ(num_qubits=num_qubits, depth=3)

        svm = QSVMVariational(optimizer, feature_map, var_form, self.training_data, self.testing_data)
        aqua_globals.random_seed = self.random_seed
        quantum_instance = QuantumInstance(backend, shots=1024, seed=self.random_seed, seed_mapper=self.random_seed)
        result = svm.run(quantum_instance)

        np.testing.assert_array_almost_equal(result['opt_params'], self.ref_opt_params, decimal=4)
        np.testing.assert_array_almost_equal(result['training_loss'], self.ref_train_loss, decimal=8)

        self.assertEqual(result['testing_accuracy'], 0.5)

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
        backend = BasicAer.get_backend('qasm_simulator')

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
                ['0', '[ 0.18863864 -1.08197582  1.74432295  1.29765602]', '0.54802', '0'],
                ['1', '[ 1.18863864 -1.08197582  1.74432295  1.29765602]', '0.53862', '1'],
                ['2', '[ 1.18863864 -0.08197582  1.74432295  1.29765602]', '0.47278', '2'],
        ]
        try:
            with open(self._get_resource_path(tmp_filename)) as f:
                idx = 0
                for record in f.readlines():
                    eval_count, parameters, cost, batch_index = record.split(",")
                    self.assertEqual(eval_count.strip(), ref_content[idx][0])
                    self.assertEqual(parameters, ref_content[idx][1])
                    self.assertEqual(cost.strip(), ref_content[idx][2])
                    self.assertEqual(batch_index.strip(), ref_content[idx][3])
                    idx += 1
        finally:
            if is_file_exist:
                os.remove(self._get_resource_path(tmp_filename))


if __name__ == '__main__':
    unittest.main()
