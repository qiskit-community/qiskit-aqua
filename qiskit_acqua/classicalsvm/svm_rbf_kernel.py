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

from sklearn.metrics.pairwise import rbf_kernel

from qiskit_acqua import QuantumAlgorithm
from qiskit_acqua.svm import (get_points_and_labels, optimize_SVM)


class SVM_RBF_Kernel(QuantumAlgorithm):
    SVM_RBF_KERNEL_CONFIGURATION = {
        'name': 'SVM_RBF_Kernel',
        'description': 'SVM_RBF_Kernel Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'SVM_RBF_schema',
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
        super().__init__(configuration or self.SVM_RBF_KERNEL_CONFIGURATION.copy())
        self._ret = {}

    def kernel_join_classical(self, points_array, points_array2, gamma=None):
        return rbf_kernel(points_array, points_array2, gamma)

    def init_params(self, params, algo_input):
        SVM_RBF_K_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        self.training_dataset = algo_input.training_dataset
        self.test_dataset = algo_input.test_dataset
        self.datapoints = algo_input.datapoints
        self.class_labels = list(self.training_dataset.keys())

        self.init_args(SVM_RBF_K_params.get('print_info'))

    def init_args(self, print_info=False):
        self.print_info = print_info

    def train(self, training_input, class_labels):
        training_points, training_points_labels, label_to_class = get_points_and_labels(training_input, class_labels)

        Kernel_mat = self.kernel_join_classical(training_points, training_points)

        if self.print_info:
            import matplotlib.pyplot as plt
            img = plt.imshow(np.asmatrix(Kernel_mat), interpolation='nearest', origin='upper', cmap='bone_r')
            plt.show()

        [alpha, b, support] = optimize_SVM(Kernel_mat, training_points_labels)
        alphas = np.array([])
        SVMs = np.array([])
        yin = np.array([])
        for alphindex in range(len(support)):
            if support[alphindex]:
                alphas = np.vstack([alphas, alpha[alphindex]]) if alphas.size else alpha[alphindex]
                SVMs = np.vstack([SVMs, training_points[alphindex]]) if SVMs.size else training_points[alphindex]
                yin = np.vstack([yin, training_points_labels[alphindex]]) if yin.size else training_points_labels[alphindex]

        return [alphas, b, SVMs, yin]

    def test(self, svm, test_input, class_labels):
        test_points, test_points_labels, label_to_labelclass = get_points_and_labels(test_input, class_labels)

        alphas, bias, SVMs, yin = svm
        kernel_matrix = self.kernel_join_classical(test_points, SVMs)
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

    def predict(self, svm, test_points):
        alphas, bias, SVMs, yin = svm
        kernel_matrix = self.kernel_join_classical(test_points, SVMs)

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

        svm = self.train(self.training_dataset, self.class_labels)

        if self.test_dataset is not None:
            success_ratio = self.test(svm, self.test_dataset, self.class_labels)
            self._ret['test_success_ratio'] = success_ratio

        if self.datapoints is not None:
            predicted_labels = self.predict(svm, self.datapoints)
            _, _, label_to_class = get_points_and_labels(self.training_dataset, self.class_labels)
            predicted_labelclasses = [label_to_class[x] for x in predicted_labels]
            self._ret['predicted_labels'] = predicted_labelclasses
        return self._ret
