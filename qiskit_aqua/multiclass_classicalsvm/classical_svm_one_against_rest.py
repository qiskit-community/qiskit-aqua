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
import copy
from qiskit_aqua import QuantumAlgorithm
from qiskit_aqua.svm import (get_points_and_labels, optimize_SVM)
from qiskit_aqua.multiclass.one_against_rest import OneAgainstRest
from qiskit_aqua.multiclass_classicalsvm.linear_svc_estimator import LinearSVC_Estimator
from qiskit_aqua.multiclass.data_preprocess import *

class ClassicalSVM_OneAgainstRest(QuantumAlgorithm):
    ClassicalSVM_OneAgainstRest_CONFIGURATION = {
        'name': 'ClassicalSVM_OneAgainstRest',
        'description': 'ClassicalSVM_OneAgainstRest Algorithm',
        'classical': True,
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'ClassicalSVM_OneAgainstRest_schema',
            'type': 'object',
            'properties': {
                'gamma': {
                    'type': ['number', 'null'],
                    'default': None
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
        super().__init__(configuration or copy.deepcopy(ClassicalSVM_OneAgainstRest.ClassicalSVM_OneAgainstRest_CONFIGURATION))
        self._ret = {}

    def init_params(self, params, algo_input):
        ClassicalSVM_OneAgainstRest_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        self.init_args(algo_input.training_dataset, algo_input.test_dataset,
                       algo_input.datapoints, ClassicalSVM_OneAgainstRest_params.get('print_info'))

    def init_args(self, training_dataset, test_dataset, datapoints, print_info=False):
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.datapoints = datapoints
        self.class_labels = list(self.training_dataset.keys())
        self.print_info = print_info


    def run(self):
        if self.training_dataset is None:
            self._ret['error'] = 'training dataset is missing! please provide it'
            return self._ret


        X_train, y_train, label_to_class = multiclass_get_points_and_labels(self.training_dataset, self.class_labels)
        X_test, y_test, label_to_class = multiclass_get_points_and_labels(self.test_dataset, self.class_labels)

        oar = OneAgainstRest(LinearSVC_Estimator)
        oar.train(X_train, y_train)
        # print()



        if self.test_dataset is not None:
            success_ratio = oar.test(X_test, y_test)
            self._ret['test_success_ratio'] = success_ratio

        if self.datapoints is not None:
            predicted_labels = oar.predict(X_test)
            predicted_labelclasses = [label_to_class[x] for x in predicted_labels]
            self._ret['predicted_labels'] = predicted_labelclasses
        return self._ret
