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

from test.common import QiskitAquaTestCase
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import (run_algorithm, get_algorithm_instance, get_optimizer_instance,
                         get_feature_map_instance, get_variational_form_instance)


class TestQSVMVariational(QiskitAquaTestCase):

    def setUp(self):
        self.random_seed = 10598
        self.training_data = {'A': np.asarray([[2.95309709, 2.51327412], [3.14159265, 4.08407045]]),
                              'B': np.asarray([[4.08407045, 2.26194671], [4.46106157, 2.38761042]])}
        self.testing_data = {'A': np.asarray([[3.83274304, 2.45044227]]),
                             'B': np.asarray([[3.89557489, 0.31415927]])}

        self.ref_opt_params = np.asarray([2.93868096, 0.76735399, 4.21845289, 4.28731786,
                                          -4.64804051, -4.0103384, 3.62083309, -3.1466139,
                                          3.36741576, 0.07314989, -1.92529824, -1.31781337,
                                          2.2547051, 7.29971351, 3.74421673, -3.74280352])
        self.ref_train_loss = 0.4999339230552529

        self.svm_input = get_input_instance('SVMInput')
        self.svm_input.training_dataset = self.training_data
        self.svm_input.test_dataset = self.testing_data

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

        self.assertEqual(result['testing_accuracy'], 0.5)

    def test_qsvm_variational_directly(self):
        np.random.seed(self.random_seed)
        svm = get_algorithm_instance("QSVM.Variational")
        svm.random_seed = self.random_seed
        svm.setup_quantum_backend(backend='qasm_simulator', shots=1024)

        optimizer = get_optimizer_instance('SPSA')
        optimizer.init_args(max_trials=10, c0=4.0, skip_calibration=True)
        optimizer.set_options(save_steps=1)
        num_qubits = 2

        feature_map = get_feature_map_instance('SecondOrderExpansion')
        feature_map.init_args(num_qubits=num_qubits, depth=2)

        var_form = get_variational_form_instance('RYRZ')
        var_form.init_args(num_qubits=num_qubits, depth=3)

        svm.init_args(self.training_data, self.testing_data, None, optimizer, feature_map, var_form)
        result = svm.run()

        np.testing.assert_array_almost_equal(result['opt_params'], self.ref_opt_params, decimal=4)
        np.testing.assert_array_almost_equal(result['training_loss'], self.ref_train_loss, decimal=8)

        self.assertEqual(result['testing_accuracy'], 0.5)
