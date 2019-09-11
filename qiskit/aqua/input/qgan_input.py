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

""" qgan input """

from qiskit.aqua import AquaError
from qiskit.aqua.input import AlgorithmInput
from qiskit.aqua.utils import convert_dict_to_json


class QGANInput(AlgorithmInput):
    """ qgan input """
    CONFIGURATION = {
        'name': 'QGANInput',
        'description': 'QGAN input',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
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
