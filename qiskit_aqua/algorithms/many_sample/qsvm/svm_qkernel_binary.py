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

import logging

from qiskit_aqua.algorithms.many_sample.qsvm import (get_points_and_labels, optimize_SVM, kernel_join)
from qiskit_aqua.algorithms.many_sample.qsvm import SVM_QKernel_ABC

logger = logging.getLogger(__name__)

class SVM_QKernel_Binary(SVM_QKernel_ABC):
    """
    the binary classifier
    """

    def __init__(self):
        self.ret = {}

    def train(self, training_input, class_labels):
        """
        train the svm
        Args:
            training_input (dict): dictionary which maps each class to the points in the class
            class_labels (list): list of classes. For example: ['A', 'B']
        """
        training_points, training_points_labels, label_to_class = get_points_and_labels(training_input, class_labels)

        kernel_matrix = kernel_join(training_points, training_points, self.entangler_map,
                                    self.coupling_map, self.initial_layout, self.shots,
                                    self._random_seed, self.num_of_qubits, self._backend)

        self.ret['kernel_matrix_training'] = kernel_matrix

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

        self.ret['svm'] = {}
        self.ret['svm']['alphas'] = alphas
        self.ret['svm']['bias'] = b
        self.ret['svm']['support_vectors'] = SVMs
        self.ret['svm']['yin'] = yin

    def test(self, test_input, class_labels):
        """
        test the svm
        Args:
            test_input (dict): dictionary which maps each class to the points in the class
            class_labels (list): list of classes. For example: ['A', 'B']
        """

        test_points, test_points_labels, label_to_labelclass = get_points_and_labels(test_input, class_labels)

        alphas = self.ret['svm']['alphas']
        bias = self.ret['svm']['bias']
        SVMs = self.ret['svm']['support_vectors']
        yin = self.ret['svm']['yin']

        kernel_matrix = kernel_join(test_points, SVMs, self.entangler_map, self.coupling_map,
                                    self.initial_layout, self.shots, self._random_seed,
                                    self.num_of_qubits, self._backend)

        self.ret['kernel_matrix_testing'] = kernel_matrix

        success_ratio = 0
        L = 0
        total_num_points = len(test_points)
        Lsign = np.zeros(total_num_points)
        for tin in range(total_num_points):
            Ltot = 0
            for sin in range(len(SVMs)):
                L = yin[sin] * alphas[sin] * kernel_matrix[tin][sin]
                Ltot += L

            Lsign[tin] = np.sign(Ltot + bias)



            logger.debug("\n=============================================")
            logger.debug('classifying' + str(test_points[tin]))
            logger.debug('Label should be ' + str(label_to_labelclass[np.int(test_points_labels[tin])]))
            logger.debug('Predicted label is ' + str(label_to_labelclass[np.int(Lsign[tin])]))
            if np.int(test_points_labels[tin]) == np.int(Lsign[tin]):
                logger.debug('CORRECT')
            else:
                logger.debug('INCORRECT')

            if Lsign[tin] == test_points_labels[tin]:
                success_ratio += 1
        final_success_ratio = success_ratio / total_num_points

        logger.debug('Classification success for this set is %s %% \n' % (100 * final_success_ratio))
        return final_success_ratio

    def predict(self, test_points):
        """
        predict using the svm
        Args:
            test_points (numpy.ndarray): the points
        """
        alphas = self.ret['svm']['alphas']
        bias = self.ret['svm']['bias']
        SVMs = self.ret['svm']['support_vectors']
        yin = self.ret['svm']['yin']

        kernel_matrix = kernel_join(test_points, SVMs, self.entangler_map, self.coupling_map,
                                    self.initial_layout, self.shots, self._random_seed,
                                    self.num_of_qubits, self._backend)

        self.ret['kernel_matrix_prediction'] = kernel_matrix

        total_num_points = len(test_points)
        Lsign = np.zeros(total_num_points)
        for tin in range(total_num_points):
            Ltot = 0
            for sin in range(len(SVMs)):
                L = yin[sin] * alphas[sin] * kernel_matrix[tin][sin]
                Ltot += L
            Lsign[tin] = np.int(np.sign(Ltot + bias))
        return Lsign

    def run(self):
        """
        put the train, test, predict together
        """
        if self.training_dataset is None:
            self.ret['error'] = 'training dataset is missing! please provide it'
            return self.ret

        num_of_qubits = self.auto_detect_qubitnum(self.training_dataset)  # auto-detect mode
        if num_of_qubits == -1:
            self.ret['error'] = 'Something wrong with the auto-detection of num_of_qubits'
            return self.ret
        if num_of_qubits != 2 and num_of_qubits != 3:
            self.ret['error'] = 'You should lower the feature size to 2 or 3 using PCA first!'
            return self.ret
        self.train(self.training_dataset, self.class_labels)
        if self.test_dataset is not None:
            success_ratio = self.test(self.test_dataset, self.class_labels)
            self.ret['test_success_ratio'] = success_ratio
        if self.datapoints is not None:
            predicted_labels = self.predict(self.datapoints)
            _, _, label_to_class = get_points_and_labels(self.training_dataset, self.class_labels)
            predicted_labelclasses = [label_to_class[x] for x in predicted_labels]
            self.ret['predicted_labels'] = predicted_labelclasses

        return self.ret
