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

from qiskit_aqua.svm_qkernel import entangler_map_creator

class SVM_QKernel_ABC(ABC):
    """
    abstract base class for the binary classifier and the multiclass classifier
    """
    def auto_detect_qubitnum(self, training_dataset):
        auto_detected_size = -1
        for key in training_dataset:
            val = training_dataset[key]
            for item in val:
                auto_detected_size = len(item)
                return auto_detected_size
        return auto_detected_size

    def init_args(self, training_dataset, test_dataset, datapoints, print_info, multiclass_alg, backend, shots, random_seed):
        self._backend = backend

        if 'statevector' in self._backend:
            raise ValueError('Selected backend  "{}" does not support measurements.'.format(self._backend))

        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.datapoints = datapoints
        self.class_labels = list(self.training_dataset.keys())
        self.num_of_qubits = self.auto_detect_qubitnum(training_dataset) # auto-detect mode
        self.entangler_map = entangler_map_creator(self.num_of_qubits)
        self.coupling_map = None
        self.initial_layout = None
        self.shots = shots
        self._random_seed = random_seed
        self.multiclass_alg = multiclass_alg
        self.print_info = print_info

    @abstractmethod
    def run(self):
        raise NotImplementedError( "Should have implemented this" )


