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

from qiskit_aqua.utils.multiclass.data_preprocess import *
from qiskit_aqua.svm_qkernel.data_preprocess import *


import numpy as np

from test.common import QiskitAquaTestCase
from qiskit_aqua import run_algorithm, get_algorithm_instance
from qiskit_aqua.input import get_input_instance


class TestSVMQKernel(QiskitAquaTestCase):

    def setUp(self):
        self.random_seed = 10598


    def test_classical_binary(self):
        training_input = {'A': np.asarray([
       [ 0.6560706 ,  0.17605998],
       [ 0.14154948,  0.06201424],
       [ 0.80202323,  0.40582692],
       [ 0.46779595,  0.39946754],
       [ 0.57660199,  0.21821317],]), 'B': np.asarray([[ 0.38857596, -0.33775802],
       [ 0.49946978, -0.48727951],
       [-0.30119743, -0.11221681],
       [-0.16479252, -0.08640519],
       [ 0.49156185, -0.3660534 ]])}

        test_input= {'A': np.asarray([[0.57483139, 0.47120732],
       [0.48372348, 0.25438544],
       [0.08791134, 0.11515506],
       [0.45988094, 0.32854319],
       [0.53015085, 0.41539212],]), 'B': np.asarray([[-0.06048935, -0.48345293],
       [-0.01065613, -0.33910828],
       [-0.17323832, -0.49535592],
       [ 0.14043268, -0.87869109],
       [-0.15046837, -0.47340207]])}

        temp = [test_input[k] for k in test_input]
        total_array = np.concatenate(temp)

        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'backend': {'name': 'local_qasm_simulator', 'shots': 1000},

            'algorithm': {
                'name': 'SVM_QKernel',
                'print_info': False
            }
        }

        algo_input = get_input_instance('SVMInput')
        algo_input.training_dataset = training_input
        algo_input.test_dataset = test_input
        algo_input.datapoints = total_array
        result = run_algorithm(params, algo_input)
        self.assertEqual(result['test_success_ratio'], 0.6)
        self.assertEqual(result['predicted_labels'], ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'A', 'A', 'A'])

    def test_classical_multiclass_one_against_all(self):
        training_input = {'A': np.asarray([[ 0.6560706 ,  0.17605998],
       [ 0.25776033,  0.47628296],
       [ 0.8690704 ,  0.70847635]]), 'B': np.asarray([[ 0.38857596, -0.33775802],
       [ 0.49946978, -0.48727951],
       [ 0.49156185, -0.3660534 ]]), 'C': np.asarray([[-0.68088231,  0.46824423],
       [-0.56167659,  0.65270294],
       [-0.82139073,  0.29941512]])}

        test_input= {'A': np.asarray([[0.57483139, 0.47120732],
       [0.48372348, 0.25438544],
       [0.48142649, 0.15931707]]), 'B': np.asarray([[-0.06048935, -0.48345293],
       [-0.01065613, -0.33910828],
       [ 0.06183066, -0.53376975]]), 'C': np.asarray([[-0.74561108,  0.27047295],
       [-0.69942965,  0.11885162],
       [-0.66489165,  0.1181712 ]])}


        temp = [test_input[k] for k in test_input]
        total_array = np.concatenate(temp)

        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'algorithm': {
                'name': 'SVM_QKernel',
                'print_info': False,
                'multiclass_alg':'one_against_all'
            },
            'backend': {'name': 'local_qasm_simulator_py', 'shots': 1024}
        }

        algo_input = get_input_instance('SVMInput')
        algo_input.training_dataset = training_input
        algo_input.test_dataset = test_input
        algo_input.datapoints = total_array

        result = run_algorithm(params, algo_input)
        self.assertEqual(result['test_success_ratio'], 0.4444444444444444)
        self.assertEqual(result['predicted_labels'], ['A', 'A', 'C', 'A', 'A', 'A', 'A', 'C', 'C'])


    def test_classical_multiclass_all_pairs(self):
        training_input = {'A': np.asarray([[ 0.6560706 ,  0.17605998],
       [ 0.25776033,  0.47628296],
       [ 0.8690704 ,  0.70847635]]), 'B': np.asarray([[ 0.38857596, -0.33775802],
       [ 0.49946978, -0.48727951],
       [ 0.49156185, -0.3660534 ]]), 'C': np.asarray([[-0.68088231,  0.46824423],
       [-0.56167659,  0.65270294],
       [-0.82139073,  0.29941512]])}

        test_input= {'A': np.asarray([[0.57483139, 0.47120732],
       [0.48372348, 0.25438544],
       [0.48142649, 0.15931707]]), 'B': np.asarray([[-0.06048935, -0.48345293],
       [-0.01065613, -0.33910828],
       [ 0.06183066, -0.53376975]]), 'C': np.asarray([[-0.74561108,  0.27047295],
       [-0.69942965,  0.11885162],
       [-0.66489165,  0.1181712 ]])}


        temp = [test_input[k] for k in test_input]
        total_array = np.concatenate(temp)

        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'algorithm': {
                'name': 'SVM_QKernel',
                'print_info': False,
                'multiclass_alg':'all_pairs'
            },
            'backend': {'name': 'local_qasm_simulator_py', 'shots': 1024}
        }

        algo_input = get_input_instance('SVMInput')
        algo_input.training_dataset = training_input
        algo_input.test_dataset = test_input
        algo_input.datapoints = total_array

        result = run_algorithm(params, algo_input)
        self.assertEqual(result['test_success_ratio'], 0.33333333333333337)
        self.assertEqual(result['predicted_labels'], ['A', 'A', 'C', 'A', 'A', 'A', 'A', 'B', 'C'])


    def test_classical_multiclass_error_correcting_code(self):
        training_input = {'A': np.asarray([[ 0.6560706 ,  0.17605998],
       [ 0.25776033,  0.47628296],
       [ 0.8690704 ,  0.70847635]]), 'B': np.asarray([[ 0.38857596, -0.33775802],
       [ 0.49946978, -0.48727951],
       [ 0.49156185, -0.3660534 ]]), 'C': np.asarray([[-0.68088231,  0.46824423],
       [-0.56167659,  0.65270294],
       [-0.82139073,  0.29941512]])}

        test_input= {'A': np.asarray([[0.57483139, 0.47120732],
       [0.48372348, 0.25438544],
       [0.48142649, 0.15931707]]), 'B': np.asarray([[-0.06048935, -0.48345293],
       [-0.01065613, -0.33910828],
       [ 0.06183066, -0.53376975]]), 'C': np.asarray([[-0.74561108,  0.27047295],
       [-0.69942965,  0.11885162],
       [-0.66489165,  0.1181712 ]])}


        temp = [test_input[k] for k in test_input]
        total_array = np.concatenate(temp)

        params = {
            'problem': {'name': 'svm_classification', 'random_seed': self.random_seed},
            'algorithm': {
                'name': 'SVM_QKernel',
                'print_info': False,
                'multiclass_alg':'error_correcting_code'
            },
            'backend': {'name': 'local_qasm_simulator_py', 'shots': 1024}

        }

        algo_input = get_input_instance('SVMInput')
        algo_input.training_dataset = training_input
        algo_input.test_dataset = test_input
        algo_input.datapoints = total_array

        result = run_algorithm(params, algo_input)
        print(result)
        self.assertEqual(result['test_success_ratio'], 0.5555555555555556)
        self.assertEqual(result['predicted_labels'], ['A', 'A', 'C', 'A', 'A', 'A', 'C', 'C', 'C'])
