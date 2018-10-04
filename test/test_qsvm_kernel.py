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

import sys
import numpy as np

from test.common import QiskitAquaTestCase
from qiskit_aqua import run_algorithm, get_algorithm_instance, get_feature_map_instance
from qiskit_aqua.input import get_input_instance


class TestQSVMKernel(QiskitAquaTestCase):

    def setUp(self):
        self.random_seed = 10598
        self.shots = 8192
        np.random.seed(self.random_seed)
        self.training_data = {'A': np.asarray([[2.95309709, 2.51327412],
                                               [3.14159265, 4.08407045]]),
                              'B': np.asarray([[4.08407045, 2.26194671],
                                               [4.46106157, 2.38761042]])}
        self.testing_data = {'A': np.asarray([[3.83274304, 2.45044227]]),
                             'B': np.asarray([[3.89557489, 0.31415927]])}

        self.ref_kernel_matrix_training = np.asarray([[1., 0.85632324, 0.1184082, 0.36523438],
                                                      [0.85632324, 1., 0.11352539, 0.45068359],
                                                      [0.1184082, 0.11352539, 1., 0.6730957],
                                                      [0.36523438, 0.45068359, 0.6730957, 1.]])

        self.ref_kernel_matrix_testing = np.asarray([[0.14892578, 0.18115234, 0.47631836, 0.14709473],
                                                     [0.33239746, 0.3782959, 0.02270508, 0.16418457]])

        self.ref_support_vectors = np.asarray([[2.95309709, 2.51327412],
                                               [3.14159265, 4.08407045],
                                               [4.08407045, 2.26194671],
                                               [4.46106157, 2.38761042]])
        self.ref_alpha = np.asarray([0.38038017, 1.46000306, 0.02371895, 1.81666428])

        self.ref_bias = np.asarray([-0.03570662])

        self.svm_input = get_input_instance('SVMInput')
        self.svm_input.training_dataset = self.training_data
        self.svm_input.test_dataset = self.testing_data

    def test_qsvm_kernel_binary_via_run_algorithm(self):
        training_input = {'A': np.asarray([[0.6560706, 0.17605998], [0.14154948, 0.06201424],
                                           [0.80202323, 0.40582692], [0.46779595, 0.39946754],
                                           [0.57660199, 0.21821317]]),
                          'B': np.asarray([[0.38857596, -0.33775802], [0.49946978, -0.48727951],
                                           [-0.30119743, -0.11221681], [-0.16479252, -0.08640519],
                                           [0.49156185, -0.3660534]])}

        test_input = {'A': np.asarray([[0.57483139, 0.47120732], [0.48372348, 0.25438544],
                                       [0.08791134, 0.11515506], [0.45988094, 0.32854319],
                                       [0.53015085, 0.41539212]]),
                      'B': np.asarray([[-0.06048935, -0.48345293], [-0.01065613, -0.33910828],
                                       [-0.17323832, -0.49535592], [0.14043268, -0.87869109],
                                       [-0.15046837, -0.47340207]])}

        temp = [test_input[k] for k in test_input]
        total_array = np.concatenate(temp)

        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'backend': {'name': 'qasm_simulator', 'shots': self.shots},
            'algorithm': {
                'name': 'QSVM.Kernel'
            }
        }

        algo_input = get_input_instance('SVMInput')
        algo_input.training_dataset = training_input
        algo_input.test_dataset = test_input
        algo_input.datapoints = total_array
        result = run_algorithm(params, algo_input)
        self.assertEqual(result['testing_accuracy'], 0.6)
        self.assertEqual(result['predicted_classes'], ['A', 'A', 'A', 'A', 'A',
                                                       'A', 'B', 'A', 'A', 'A'])

    def test_qsvm_kernel_binary_directly(self):
        svm = get_algorithm_instance("QSVM.Kernel")
        svm.random_seed = self.random_seed
        svm.setup_quantum_backend(backend='qasm_simulator_py', shots=self.shots)

        num_qubits = 2
        feature_map = get_feature_map_instance('SecondOrderExpansion')
        feature_map.init_args(num_qubits=num_qubits, depth=2, entangler_map={0: [1]})
        svm.init_args(self.training_data, self.testing_data, None, feature_map, None)

        result = svm.run()
        np.testing.assert_array_almost_equal(
            result['kernel_matrix_training'], self.ref_kernel_matrix_training, decimal=4)
        np.testing.assert_array_almost_equal(
            result['kernel_matrix_testing'], self.ref_kernel_matrix_testing, decimal=4)

        self.assertEqual(len(result['svm']['support_vectors']), 4)
        np.testing.assert_array_almost_equal(
            result['svm']['support_vectors'], self.ref_support_vectors, decimal=4)

        np.testing.assert_array_almost_equal(result['svm']['alphas'], self.ref_alpha, decimal=4)
        np.testing.assert_array_almost_equal(result['svm']['bias'], self.ref_bias, decimal=4)

        self.assertEqual(result['testing_accuracy'], 0.5)

    def test_qsvm_kernel_binary_directly_statevector(self):
        svm = get_algorithm_instance("QSVM.Kernel")
        svm.random_seed = self.random_seed
        svm.setup_quantum_backend(backend='statevector_simulator')

        num_qubits = 2
        feature_map = get_feature_map_instance('SecondOrderExpansion')
        feature_map.init_args(num_qubits=num_qubits, depth=2, entangler_map={0: [1]})
        svm.init_args(self.training_data, self.testing_data, None, feature_map, None)

        result = svm.run()

        self.assertEqual(len(result['svm']['support_vectors']), 4)
        np.testing.assert_array_almost_equal(
            result['svm']['support_vectors'], self.ref_support_vectors, decimal=4)

        self.assertEqual(result['testing_accuracy'], 0.5)

    def test_qsvm_kernel_multiclass_one_against_all(self):
        training_input = {'A': np.asarray([[0.6560706, 0.17605998], [0.25776033, 0.47628296],
                                           [0.8690704, 0.70847635]]),
                          'B': np.asarray([[0.38857596, -0.33775802], [0.49946978, -0.48727951],
                                           [0.49156185, -0.3660534]]),
                          'C': np.asarray([[-0.68088231, 0.46824423], [-0.56167659, 0.65270294],
                                           [-0.82139073, 0.29941512]])}

        test_input = {'A': np.asarray([[0.57483139, 0.47120732], [0.48372348, 0.25438544],
                                       [0.48142649, 0.15931707]]),
                      'B': np.asarray([[-0.06048935, -0.48345293], [-0.01065613, -0.33910828],
                                       [0.06183066, -0.53376975]]),
                      'C': np.asarray([[-0.74561108, 0.27047295], [-0.69942965, 0.11885162],
                                       [-0.66489165, 0.1181712]])}

        temp = [test_input[k] for k in test_input]
        total_array = np.concatenate(temp)

        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'algorithm': {
                'name': 'QSVM.Kernel',
            },
            'backend': {'name': 'qasm_simulator', 'shots': self.shots},
            'multiclass_extension': {'name': 'OneAgainstRest'},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'entangler_map': {0: [1]}}
        }

        algo_input = get_input_instance('SVMInput')
        algo_input.training_dataset = training_input
        algo_input.test_dataset = test_input
        algo_input.datapoints = total_array

        result = run_algorithm(params, algo_input)

        # Note: Result here is dependent on platform with the C++ simulator
        expected_accuracy = 0.555555555 if sys.platform.startswith('linux') else 0.444444444
        expected_classes = ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'C'] if sys.platform.startswith('linux') else \
                           ['A', 'A', 'C', 'A', 'A', 'A', 'A', 'C', 'C']
        self.assertAlmostEqual(result['testing_accuracy'], expected_accuracy, places=4,
                               msg='Please ensure you are using c++ simulator')
        self.assertEqual(result['predicted_classes'], expected_classes)

    def test_qsvm_kernel_multiclass_all_pairs(self):
        training_input = {'A': np.asarray([[0.6560706, 0.17605998], [0.25776033, 0.47628296],
                                           [0.8690704, 0.70847635]]),
                          'B': np.asarray([[0.38857596, -0.33775802], [0.49946978, -0.48727951],
                                           [0.49156185, -0.3660534]]),
                          'C': np.asarray([[-0.68088231, 0.46824423], [-0.56167659, 0.65270294],
                                           [-0.82139073, 0.29941512]])}

        test_input = {'A': np.asarray([[0.57483139, 0.47120732], [0.48372348, 0.25438544],
                                       [0.48142649, 0.15931707]]),
                      'B': np.asarray([[-0.06048935, -0.48345293], [-0.01065613, -0.33910828],
                                       [0.06183066, -0.53376975]]),
                      'C': np.asarray([[-0.74561108, 0.27047295], [-0.69942965, 0.11885162],
                                       [-0.66489165, 0.1181712]])}

        temp = [test_input[k] for k in test_input]
        total_array = np.concatenate(temp)

        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'algorithm': {
                'name': 'QSVM.Kernel',
            },
            'backend': {'name': 'qasm_simulator', 'shots': self.shots},
            'multiclass_extension': {'name': 'AllPairs'},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'entangler_map': {0: [1]}}
        }

        algo_input = get_input_instance('SVMInput')
        algo_input.training_dataset = training_input
        algo_input.test_dataset = test_input
        algo_input.datapoints = total_array
        result = run_algorithm(params, algo_input)
        self.assertAlmostEqual(result['testing_accuracy'], 0.444444444, places=4,
                               msg='Please ensure you are using c++ simulator')
        self.assertEqual(result['predicted_classes'], ['A', 'A', 'C', 'A',
                                                       'A', 'A', 'A', 'C', 'C'])

    def test_qsvm_kernel_multiclass_error_correcting_code(self):
        training_input = {'A': np.asarray([[0.6560706, 0.17605998], [0.25776033, 0.47628296],
                                           [0.8690704, 0.70847635]]),
                          'B': np.asarray([[0.38857596, -0.33775802], [0.49946978, -0.48727951],
                                           [0.49156185, -0.3660534]]),
                          'C': np.asarray([[-0.68088231, 0.46824423], [-0.56167659, 0.65270294],
                                           [-0.82139073, 0.29941512]])}

        test_input = {'A': np.asarray([[0.57483139, 0.47120732], [0.48372348, 0.25438544],
                                       [0.48142649, 0.15931707]]),
                      'B': np.asarray([[-0.06048935, -0.48345293], [-0.01065613, -0.33910828],
                                       [0.06183066, -0.53376975]]),
                      'C': np.asarray([[-0.74561108, 0.27047295], [-0.69942965, 0.11885162],
                                       [-0.66489165, 0.1181712]])}

        temp = [test_input[k] for k in test_input]
        total_array = np.concatenate(temp)

        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'algorithm': {
                'name': 'QSVM.Kernel',
            },
            'backend': {'name': 'qasm_simulator', 'shots': 8192},
            'multiclass_extension': {'name': 'ErrorCorrectingCode', 'code_size': 5},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'entangler_map': {0: [1]}}
        }

        algo_input = get_input_instance('SVMInput')
        algo_input.training_dataset = training_input
        algo_input.test_dataset = test_input
        algo_input.datapoints = total_array

        result = run_algorithm(params, algo_input)
        self.assertAlmostEqual(result['testing_accuracy'], 0.55555555, places=4,
                               msg='Please ensure you are using c++ simulator')
        self.assertEqual(result['predicted_classes'], ['A', 'A', 'C', 'A',
                                                       'A', 'A', 'C', 'C', 'C'])
