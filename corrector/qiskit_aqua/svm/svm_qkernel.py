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

import numpy as np

from qiskit_aqua import QuantumAlgorithm
from qiskit_aqua.svm import (get_points_and_labels, optimize_SVM,
                             kernel_join, entangler_map_creator)


class SVM_QKernel(QuantumAlgorithm):
    SVM_QKERNEL_CONFIGURATION = {
        'name': 'SVM_QKernel',
        'description': 'SVM_QKernel Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'SVM_QKernel_schema',
            'type': 'object',
            'properties': {
                'print_info': {
                    'type': 'boolean',
                    'default': False
                }
            },
            'additionalProperties': False
        },
        'problems': ['svm_classification']
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.SVM_QKERNEL_CONFIGURATION.copy())
        self._ret = {}

    def init_params(self, params, algo_input):
        SVMQK_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)

        self.init_args(algo_input.training_dataset, algo_input.test_dataset,
                       algo_input.datapoints, SVMQK_params.get('print_info'))

    def auto_detect_qubitnum(self, training_dataset):
        auto_detected_size = -1
        for key in training_dataset:
            val = training_dataset[key]
            for item in val:
                auto_detected_size = len(item)
                return auto_detected_size
        return auto_detected_size

    def init_args(self, training_dataset, test_dataset, datapoints, print_info=False):  # 2
        if 'statevector' in self._backend:
            raise ValueError('Selected backend  "{}" does not support measurements.'.format(self._backend))

        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.datapoints = datapoints
        self.class_labels = class_labels = list(self.training_dataset.keys())

        self.num_of_qubits = self.auto_detect_qubitnum(training_dataset) # auto-detect mode
        self.entangler_map = entangler_map_creator(self.num_of_qubits)
        self.coupling_map = None
        self.initial_layout = None
        self.shots = self._execute_config['shots']

        self.print_info = print_info

    def train(self, training_input, class_labels):
        training_points, training_points_labels, label_to_class = get_points_and_labels(training_input, class_labels)

        kernel_matrix = kernel_join(training_points, training_points, self.entangler_map,
                                    self.coupling_map, self.initial_layout, self.shots,
                                    self._random_seed, self.num_of_qubits, self._backend)

        self._ret['kernel_matrix_training'] = kernel_matrix

        [alpha, b, support] = optimize_SVM(kernel_matrix, training_points_labels)
        alphas = np.array([])
        SVMs = np.array([])
        yin = np.array([])
        for alphindex in range(len(support)):
            if support[alphindex]:
                alphas = np.vstack([alphas, alpha[alphindex]]) if alphas.size else alpha[alphindex]
                SVMs = np.vstack([SVMs, training_points[alphindex]]) if SVMs.size else training_points[alphindex]
                yin = np.vstack([yin, training_points_labels[alphindex]]
                                ) if yin.size else training_points_labels[alphindex]

        self._ret['svm'] = {}
        self._ret['svm']['alphas'] = alphas
        self._ret['svm']['bias'] = b
        self._ret['svm']['support_vectors'] = SVMs
        self._ret['svm']['yin'] = yin

    def test(self, test_input, class_labels):
        test_points, test_points_labels, label_to_labelclass = get_points_and_labels(test_input, class_labels)

        alphas = self._ret['svm']['alphas']
        bias = self._ret['svm']['bias']
        SVMs = self._ret['svm']['support_vectors']
        yin = self._ret['svm']['yin']

        kernel_matrix = kernel_join(test_points, SVMs, self.entangler_map, self.coupling_map,
                                    self.initial_layout, self.shots, self._random_seed,
                                    self.num_of_qubits, self._backend)

        self._ret['kernel_matrix_testing'] = kernel_matrix

        success_ratio = 0
        L = 0
        total_num_points = len(test_points)
        Lsign = np.zeros(total_num_points)
        for tin in range(total_num_points):
            Ltot = 0
            for sin in range(len(SVMs)):
                L = yin[sin]*alphas[sin]*kernel_matrix[tin][sin]
                Ltot += L

            Lsign[tin] = np.sign(Ltot+bias)
            if self.print_info:
                print("\n=============================================")
                print('classifying', test_points[tin])
                print('Label should be ', label_to_labelclass[np.int(test_points_labels[tin])])
                print('Predicted label is ', label_to_labelclass[np.int(Lsign[tin])])
                if np.int(test_points_labels[tin]) == np.int(Lsign[tin]):
                    print('CORRECT')
                else:
                    print('INCORRECT')

            if Lsign[tin] == test_points_labels[tin]:
                success_ratio += 1
        final_success_ratio = success_ratio/total_num_points
        if self.print_info:
            print('Classification success for this set is %s %% \n' % (100*final_success_ratio))
        return final_success_ratio

    def predict(self, test_points):

        alphas = self._ret['svm']['alphas']
        bias = self._ret['svm']['bias']
        SVMs = self._ret['svm']['support_vectors']
        yin = self._ret['svm']['yin']

        kernel_matrix = kernel_join(test_points, SVMs, self.entangler_map, self.coupling_map,
                                    self.initial_layout, self.shots, self._random_seed,
                                    self.num_of_qubits, self._backend)

        self._ret['kernel_matrix_prediction'] = kernel_matrix

        total_num_points = len(test_points)
        Lsign = np.zeros(total_num_points)
        for tin in range(total_num_points):
            Ltot = 0
            for sin in range(len(SVMs)):
                L = yin[sin]*alphas[sin]*kernel_matrix[tin][sin]
                Ltot += L
            Lsign[tin] = np.int(np.sign(Ltot+bias))
        return Lsign

    def run(self):
        if self.training_dataset is None:
            self._ret['error'] = 'training dataset is missing! please provide it'
            return self._ret

        num_of_qubits = self.auto_detect_qubitnum(self.training_dataset) # auto-detect mode
        if num_of_qubits == -1:
            self._ret['error'] = 'Something wrong with the auto-detection of num_of_qubits'
            return self._ret
        if num_of_qubits != 2 and num_of_qubits != 3:
            self._ret['error'] = 'You should lower the feature size to 2 or 3 using PCA first!'
            return self._ret


        self.train(self.training_dataset, self.class_labels)

        if self.test_dataset is not None:
            success_ratio = self.test(self.test_dataset, self.class_labels)
            self._ret['test_success_ratio'] = success_ratio

        if self.datapoints is not None:
            predicted_labels = self.predict(self.datapoints)
            _, _, label_to_class = get_points_and_labels(self.training_dataset, self.class_labels)
            predicted_labelclasses = [label_to_class[x] for x in predicted_labels]
            self._ret['predicted_labels'] = predicted_labelclasses

        return self._ret
