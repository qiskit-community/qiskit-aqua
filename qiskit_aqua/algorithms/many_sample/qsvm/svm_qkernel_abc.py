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


from abc import ABC, abstractmethod

import numpy as np

from qiskit_aqua import AlgorithmError

# from qiskit_aqua.algorithms.many_sample.qsvm import entangler_map_creator


class SVM_QKernel_ABC(ABC):
    """
    abstract base class for the binary classifier and the multiclass classifier
    """

    def __init__(self):
        self._ret = {}

    def auto_detect_qubitnum(self, training_dataset):
        auto_detected_size = -1
        for key in training_dataset:
            val = training_dataset[key]
            for item in val:
                auto_detected_size = len(item)
                return auto_detected_size
        return auto_detected_size

    def init_args(self, training_dataset, test_dataset, datapoints, feature_map, qalgo):

        if training_dataset is None:
            raise ValueError('training dataset is missing! please provide it')

        self.class_labels = list(training_dataset.keys())
        self.label_to_class = {idx: name for idx, name in enumerate(self.class_labels)}
        self.class_to_label = {name: idx for idx, name in enumerate(self.class_labels)}
        self.num_classes = len(self.class_labels)

        self.training_dataset = self._split_dataset_to_data_and_labels(training_dataset,
                                                                       self.class_to_label)
        self.training_dataset_all = training_dataset
        self.test_dataset_all = test_dataset
        if test_dataset is not None:
            self.test_dataset = self._split_dataset_to_data_and_labels(test_dataset,
                                                                       self.class_to_label)
        self.datapoints = datapoints
        self.feature_map = feature_map
        self.num_qubits = self.feature_map.num_qubits
        self.qalgo = qalgo

        # if len(self.class_labels) > 2 and self.multiclass_alg is None:
        #     raise AlgorithmError('For multiclass problem, please select multiclass algorithm.')

    @staticmethod
    def _split_dataset_to_data_and_labels(dataset, class_to_label):
        """Split dataset to data and labels numpy array

        Args:
            dataset (dict):

        Returns:
            numpy.ndarray: data, NxD array
            numpy.ndarray: labels, Nx1 array
        """
        data = []
        labels = []
        for key, values in dataset.items():
            for value in values:
                data.append(value)
                try:
                    labels.append(class_to_label[key])
                except Exception as e:
                    raise AlgorithmError('The dataset has different class names to '
                                         'the training data. error message: {}'.format(e))
        data = np.asarray(data)
        labels = np.asarray(labels)
        return [data, labels]

    def label_to_class_name(self, predicted_labels):
        """Helper converts labels (numeric) to class name (string)
        Args:
            predicted_labels (numpy.ndarray): Nx1 array
        Returns:
            [str]: predicted class names of each datum
        """

        if not isinstance(predicted_labels, np.ndarray):
            predicted_labels = np.asarray([predicted_labels])

        predicted_class_names = []

        for predicted_label in predicted_labels:
            predicted_class_names.append(self.label_to_class[predicted_label])
        return predicted_class_names

    @abstractmethod
    def run(self):
        raise NotImplementedError("Should have implemented this")

    @property
    def ret(self):
        return self._ret
