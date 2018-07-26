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

from test.common import QiskitAquaTestCase
from qiskit_aqua import Operator, run_algorithm
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import get_algorithm_instance


class TestSVMQKernel(QiskitAquaTestCase):

    def setUp(self):
        self.random_seed = 10598
        self.training_data = {'A': np.asarray([[2.95309709, 2.51327412], [3.14159265, 4.08407045]]),
                              'B': np.asarray([[4.08407045, 2.26194671], [4.46106157, 2.38761042]])}
        self.testing_data = {'A': np.asarray([[3.83274304, 2.45044227]]),
                             'B': np.asarray([[3.89557489, 0.31415927]])}

        self.ref_kernel_matrix_training = np.asarray([[1., 0.8388671875, 0.1142578125, 0.3564453125],
                                                      [0.8388671875, 1., 0.1044921875, 0.427734375],
                                                      [0.1142578125, 0.1044921875, 1., 0.66015625],
                                                      [0.3564453125, 0.427734375, 0.66015625, 1.]])

        self.ref_kernel_matrix_testing = np.asarray([[0.1357421875, 0.166015625, 0.4580078125, 0.140625],
                                                     [0.3076171875, 0.3623046875, 0.0166015625, 0.150390625]])

        self.ref_support_vectors = np.asarray([[2.95309709, 2.51327412], [3.14159265, 4.08407045],
                                               [4.08407045, 2.26194671], [4.46106157, 2.38761042]])

        self.svm_input = get_input_instance('SVMInput')
        self.svm_input.training_dataset = self.training_data
        self.svm_input.test_dataset = self.testing_data

    def test_svm_qkernel_via_run_algorithm(self):

        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'algorithm': {'name': 'SVM_QKernel'},
            'backend': {'name': 'local_qasm_simulator', 'shots': 1024}
        }
        result = run_algorithm(params, self.svm_input)

        np.testing.assert_array_almost_equal(
            result['kernel_matrix_training'], self.ref_kernel_matrix_training, decimal=4)
        np.testing.assert_array_almost_equal(
            result['kernel_matrix_testing'], self.ref_kernel_matrix_testing, decimal=4)

        self.assertEqual(len(result['svm']['support_vectors']), 4)
        np.testing.assert_array_almost_equal(
            result['svm']['support_vectors'], self.ref_support_vectors, decimal=4)

        self.assertEqual(result['test_success_ratio'], 0.5)

    def test_svm_qkernel_directly(self):
        svm = get_algorithm_instance("SVM_QKernel")
        svm.setup_quantum_backend(backend='local_qasm_simulator', shots=1024)
        svm.random_seed = self.random_seed
        svm.init_args(self.training_data, self.testing_data, None, print_info=False)
        result = svm.run()

        np.testing.assert_array_almost_equal(
            result['kernel_matrix_training'], self.ref_kernel_matrix_training, decimal=4)
        np.testing.assert_array_almost_equal(
            result['kernel_matrix_testing'], self.ref_kernel_matrix_testing, decimal=4)

        self.assertEqual(len(result['svm']['support_vectors']), 4)
        np.testing.assert_array_almost_equal(
            result['svm']['support_vectors'], self.ref_support_vectors, decimal=4)

        self.assertEqual(result['test_success_ratio'], 0.5)
