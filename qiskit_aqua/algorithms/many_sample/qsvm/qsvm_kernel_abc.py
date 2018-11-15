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

from qiskit_aqua.utils import split_dataset_to_data_and_labels


class QSVM_Kernel_ABC(ABC):
    """
    abstract base class for the binary classifier and the multiclass classifier
    """

    def __init__(self):
        self._ret = {}

    def init_args(self, training_dataset, test_dataset, datapoints, feature_map, qalgo):

        if training_dataset is None:
            raise ValueError('training dataset is missing! please provide it')

        self.training_dataset, self.class_to_label = split_dataset_to_data_and_labels(
            training_dataset)
        if test_dataset is not None:
            self.test_dataset = split_dataset_to_data_and_labels(test_dataset,
                                                                 self.class_to_label)

        self.label_to_class = {label: class_name for class_name, label
                               in self.class_to_label.items()}
        self.num_classes = len(list(self.class_to_label.keys()))

        self.datapoints = datapoints
        self.feature_map = feature_map
        self.num_qubits = self.feature_map.num_qubits
        self.qalgo = qalgo

    @abstractmethod
    def run(self):
        raise NotImplementedError("Should have implemented this")

    @property
    def ret(self):
        return self._ret
