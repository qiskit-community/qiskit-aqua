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

import logging

# from qiskit_aqua.algorithms.components.multiclass import multiclass_get_points_and_labels
from qiskit_aqua.algorithms.many_sample.qsvm import SVM_QKernel_ABC

logger = logging.getLogger(__name__)

class SVM_QKernel_Multiclass(SVM_QKernel_ABC):
    """
    the multiclass classifier
    the classifier is built by wrapping the estimator (for binary classification) with the multiclass extensions
    """

    def __init__(self, multiclass_classifier):
        # self.ret = {}
        super().__init__()
        self.multiclass_classifier = multiclass_classifier

    def train(self, data, labels):
        # x_train, y_train, label_to_class = multiclass_get_points_and_labels(training_dataset, class_labels)
        self.multiclass_classifier.train(data, labels)

    def test(self, data, labels):
        # x_test, y_test, label_to_class = multiclass_get_points_and_labels(test_dataset, class_labels)
        success_ratio = self.multiclass_classifier.test(data, labels)
        self._ret['testing_accuracy'] = success_ratio
        self._ret['test_success_ratio'] = success_ratio

    def predict(self, data):
        predicted_labels = self.multiclass_classifier.predict(data)
        predicted_classes = self.label_to_class_name(predicted_labels)
        # predicted_labelclasses = [label_to_class[x] for x in predicted_labels]
        self._ret['predicted_labels'] = predicted_classes

        return predicted_classes

    def run(self):
        """
        put the train, test, predict together
        """
        self.train(self.training_dataset[0], self.training_dataset[1])
        if self.test_dataset is not None:
            self.test(self.test_dataset[0], self.test_dataset[1])
        if self.datapoints is not None:
            self.predict(self.datapoints)

        return self._ret
