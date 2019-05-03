# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging

import numpy as np

from qiskit.aqua.algorithms.classical.svm import _SVM_Classical_ABC
from qiskit.aqua.utils import map_label_to_class_name

logger = logging.getLogger(__name__)


class _SVM_Classical_Multiclass(_SVM_Classical_ABC):
    """
    the multiclass classifier
    the classifier is built by wrapping the estimator
    (for binary classification) with the multiclass extensions
    """

    def __init__(self, training_dataset, test_dataset, datapoints, gamma, multiclass_classifier):
        super().__init__(training_dataset, test_dataset, datapoints, gamma)
        self.multiclass_classifier = multiclass_classifier

    def train(self, data, labels):
        self.multiclass_classifier.train(data, labels)

    def test(self, data, labels):
        accuracy = self.multiclass_classifier.test(data, labels)
        self._ret['testing_accuracy'] = accuracy
        self._ret['test_success_ratio'] = accuracy
        return accuracy

    def predict(self, data):
        predicted_labels = self.multiclass_classifier.predict(data)
        self._ret['predicted_labels'] = predicted_labels
        return predicted_labels

    def run(self):
        """
        put the train, test, predict together
        """
        self.train(self.training_dataset[0], self.training_dataset[1])
        if self.test_dataset is not None:
            self.test(self.test_dataset[0], self.test_dataset[1])
        if self.datapoints is not None:
            predicted_labels = self.predict(self.datapoints)
            predicted_classes = map_label_to_class_name(predicted_labels, self.label_to_class)
            self._ret['predicted_classes'] = predicted_classes

        return self._ret

    def load_model(self, file_path):
        model_npz = np.load(file_path)
        for i in range(len(self.multiclass_classifier.estimators)):
            self.multiclass_classifier.estimators.ret['svm']['alphas'] = model_npz['alphas_{}'.format(i)]
            self.multiclass_classifier.estimators.ret['svm']['bias'] = model_npz['bias_{}'.format(i)]
            self.multiclass_classifier.estimators.ret['svm']['support_vectors'] = model_npz['support_vectors_{}'.format(i)]
            self.multiclass_classifier.estimators.ret['svm']['yin'] = model_npz['yin_{}'.format(i)]

    def save_model(self, file_path):
        model = {}
        for i, estimator in enumerate(self.multiclass_classifier.estimators):
            model['alphas_{}'.format(i)] = estimator.ret['svm']['alphas']
            model['bias_{}'.format(i)] = estimator.ret['svm']['bias']
            model['support_vectors_{}'.format(i)] = estimator.ret['svm']['support_vectors']
            model['yin_{}'.format(i)] = estimator.ret['svm']['yin']
        np.savez(file_path, **model)
