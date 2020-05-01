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

""" Classification Input """

from qiskit.aqua import AquaError
from qiskit.aqua.input import AlgorithmInput
from qiskit.aqua.utils import convert_dict_to_json


class ClassificationInput(AlgorithmInput):
    """ Classification Input """
    CONFIGURATION = {
        'name': 'ClassificationInput',
        'description': 'SVM input',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'classification_input_schema',
            'type': 'object',
            'properties': {
                'training_dataset': {
                    'type': ['object', 'null'],
                    'default': None
                },
                'test_dataset': {
                    'type': ['object', 'null'],
                    'default': None
                },
                'datapoints': {
                    'type': ['array', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        },
        'problems': ['classification']
    }

    def __init__(self, training_dataset, test_dataset=None, datapoints=None):
        self.validate(locals())
        super().__init__()
        self.training_dataset = training_dataset or {}
        self.test_dataset = test_dataset or {}
        self.datapoints = datapoints if datapoints is not None else []

    def validate(self, args_dict):
        params = {key: value for key, value in args_dict.items()
                  if key in ['training_dataset', 'test_dataset', 'datapoints']}
        super().validate(convert_dict_to_json(params))

    def to_params(self):
        params = {}
        params['training_dataset'] = self.training_dataset
        params['test_dataset'] = self.test_dataset
        params['datapoints'] = self.datapoints
        return params

    @classmethod
    def from_params(cls, params):
        if 'training_dataset' not in params:
            raise AquaError("training_dataset is required.")
        training_dataset = params['training_dataset']
        test_dataset = params['test_dataset']
        datapoints = params['datapoints']
        return cls(training_dataset, test_dataset, datapoints)
