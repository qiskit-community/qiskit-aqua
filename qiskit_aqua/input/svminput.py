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

import copy

from qiskit_aqua import AlgorithmError
from qiskit_aqua.input import AlgorithmInput


class SVMInput(AlgorithmInput):

    SVM_INPUT_CONFIGURATION = {
        'name': 'SVMInput',
        'description': 'SVM input',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'svm_input_schema',
            'type': 'object',
            'properties': {
                'training_dataset':{
                    'type': 'object'
                },
                'test_dataset':{
                    'type': 'object'
                },
                'datapoints':{
                    'type': 'array'
                }
            },
            'additionalProperties': False
        },
        'problems': ['svm_classification']
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or copy.deepcopy(self.SVM_INPUT_CONFIGURATION))
        self.training_dataset = None
        self.test_dataset = None
        self.datapoints = None




    def to_params(self):
        params = {}
        params['training_dataset'] = self.training_dataset
        params['test_dataset'] = self.test_dataset
        params['datapoints'] = self.datapoints
        return params

    def from_params(self, params):
        if 'training_dataset' not in params:
            raise AlgorithmError("training_dataset is required.")
        self.training_dataset = params['training_dataset']
        self.test_dataset = params['test_dataset']
        self.datapoints = params['datapoints']
