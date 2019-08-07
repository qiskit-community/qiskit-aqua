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

from qiskit.aqua import AquaError
from qiskit.aqua.input import AlgorithmInput
from qiskit.aqua.utils import convert_dict_to_json


class QGANInput(AlgorithmInput):

    CONFIGURATION = {
        'name': 'QGANInput',
        'description': 'QGAN input',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'qgan_input_schema',
            'type': 'object',
            'properties': {
                'data': {
                    'type': ['array', 'null'],
                    'default': None
                },
                'bounds': {
                    'type': ['array', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        },
        'problems': ['distribution_learning_loading']
    }

    def __init__(self, data, bounds):
        self.validate(locals())
        super().__init__()
        self.data = data
        self.bounds = bounds

    def validate(self, args_dict):
        params = {key: value for key, value in args_dict.items() if key in ['data', 'bounds']}
        super().validate(convert_dict_to_json(params))

    def to_params(self):
        params = {}
        params['data'] = self.data
        params['bounds'] = self.bounds
        return params

    @classmethod
    def from_params(cls, params):
        if 'data' not in params:
            raise AquaError("Training data not given.")
        if 'bounds' not in params:
            raise AquaError("Data bounds not given.")
        data = params['data']
        bounds = params['bounds']
        return cls(data, bounds)
