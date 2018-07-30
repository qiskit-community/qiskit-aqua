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

from qiskit_aqua.multiclass.multiclass_quantumsvm.qkernel_svm_estimator import QKernalSVM_Estimator
import numpy as np
from qiskit_aqua.multiclass.data_preprocess import *
from qiskit_aqua.multiclass.error_correcting_code import ErrorCorrectingCode
from qiskit_aqua.multiclass.allpairs import AllPairs
from qiskit_aqua.multiclass.one_against_rest import OneAgainstRest


class SVM_Quantum_Multiclass(QuantumAlgorithm):
    SVM_Quantum_Multiclass_CONFIGURATION = {
        'name': 'SVM_Quantum_Multiclass',
        'description': 'SVM_Quantum_Multiclass Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'SVM_Quantum_Multiclass_schema',
            'type': 'object',
            'properties': {
                'multiclass_alg': {
                    'type': 'string',
                    'default': 'all_pairs'
                },
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
        super().__init__(configuration or self.SVM_Quantum_Multiclass_CONFIGURATION.copy())
        self._ret = {}

    def init_params(self, params, algo_input):
        svm_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)

        self.init_args(algo_input.training_dataset, algo_input.test_dataset,
                       algo_input.datapoints, svm_params.get('print_info'), svm_params.get('multiclass_alg'))

    def auto_detect_qubitnum(self, training_dataset):
        auto_detected_size = -1
        for key in training_dataset:
            val = training_dataset[key]
            for item in val:
                auto_detected_size = len(item)
                return auto_detected_size
        return auto_detected_size

    def init_args(self, training_dataset, test_dataset, datapoints, print_info, multiclass_alg):  # 2
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
        self.multiclass_alg = multiclass_alg



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

        X_train, y_train, label_to_class = multiclass_get_points_and_labels(self.training_dataset, self.class_labels)
        X_test, y_test, label_to_class = multiclass_get_points_and_labels(self.test_dataset, self.class_labels)

        if self.multiclass_alg == "all_pairs":
            multiclass_classifier = AllPairs(QKernalSVM_Estimator, [self._backend, self.shots])
        elif self.multiclass_alg == "one_against_all":
            multiclass_classifier = OneAgainstRest(QKernalSVM_Estimator, [self._backend, self.shots])
        elif self.multiclass_alg == "error_correcting_code":
            multiclass_classifier = ErrorCorrectingCode(QKernalSVM_Estimator, code_size=4, params = [self._backend, self.shots])
        else:
            self._ret['error'] = 'the multiclass alg should be one of {"all_pairs", "one_against_all", "error_correcting_code"}. You did not specify it correctly!'
            return self._ret
        if self.print_info:
            print("You are using the multiclass alg: " + self.multiclass_alg)
        multiclass_classifier.train(X_train, y_train)


        if self.test_dataset is not None:
            success_ratio = multiclass_classifier.test(X_test, y_test)
            self._ret['test_success_ratio'] = success_ratio

        if self.datapoints is not None:
            predicted_labels = multiclass_classifier.predict(X_test)
            predicted_labelclasses = [label_to_class[x] for x in predicted_labels]
            self._ret['predicted_labels'] = predicted_labelclasses

        return self._ret
