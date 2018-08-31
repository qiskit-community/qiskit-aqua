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

from qiskit_aqua.algorithms.classical.svm import SVM_Classical_ABC
from qiskit_aqua.utils import map_label_to_class_name

logger = logging.getLogger(__name__)


class SVM_Classical_Multiclass(SVM_Classical_ABC):
    """
    the multiclass classifier
    the classifier is built by wrapping the estimator
    (for binary classification) with the multiclass extensions
    """

    def __init__(self, multiclass_classifier):
        super().__init__()
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
