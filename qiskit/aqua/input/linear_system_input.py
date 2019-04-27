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

import numpy as np


class LinearSystemInput(AlgorithmInput):

    PROP_KEY_MATRIX = 'matrix'
    PROP_KEY_VECTOR = 'vector'

    CONFIGURATION = {
        'name': 'LinearSystemInput',
        'description': 'Linear System problem input',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'linear_system_state_schema',
            'type': 'object',
            'properties': {
                PROP_KEY_MATRIX: {
                    'type': ['array', 'null'],
                    'default': None
                },
                PROP_KEY_VECTOR: {
                    'type': ['array', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        },
        'problems': ['linear_system']
    }

    def __init__(self, matrix=None, vector=None):
        super().__init__()
        self._matrix = matrix if matrix is not None else []
        self._vector = vector if vector is not None else []

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        self._matrix = matrix

    @property
    def vector(self):
        return self._vector
    
    @vector.setter
    def vector(self, vector):
        self._vector = vector

    def validate(self, args_dict):
        params = {}
        for key, value in args_dict.items():
            if key == LinearSystemInput.PROP_KEY_MATRIX:
                value = value.save_to_list() if value is not None else {}
            elif key == LinearSystemInput.PROP_KEY_VECTOR:
                value = value.save_to_list() if value is not None else {}

            params[key] = value

        super().validate(params)

    def to_params(self):
        params = {}
        params[LinearSystemInput.PROP_KEY_MATRIX] = self.save_to_list(self._matrix)
        params[LinearSystemInput.PROP_KEY_VECTOR] = self.save_to_list(self._vector)
        return params

    @classmethod
    def from_params(cls, params):
        if LinearSystemInput.PROP_KEY_MATRIX not in params:
            raise AquaError("Matrix is required.")
        if LinearSystemInput.PROP_KEY_VECTOR not in params:
            raise AquaError("Vector is required.")
        mat_params = params[LinearSystemInput.PROP_KEY_MATRIX]
        matrix = cls.load_mat_from_list(mat_params)
        vec_params = params[LinearSystemInput.PROP_KEY_VECTOR]
        vector = cls.load_vec_from_list(vec_params)
        return cls(matrix, vector)

    @staticmethod
    def load_mat_from_list(mat):
        depth = lambda l: isinstance(l, list) and max(map(depth, l))+1
        if depth(mat) == 3:
            return np.array(mat[0])+1j*np.array(mat[1])
        elif depth(mat) == 2:
            return np.array(mat)
        else:
            raise AquaError("Matrix list must be depth 2 or 3")

    @staticmethod
    def load_vec_from_list(vec):
        depth = lambda l: isinstance(l, list) and max(map(depth, l))+1
        if depth(vec) == 2:
            return np.array(vec[0])+1j*np.array(vec[1])
        elif depth(vec) == 1:
            return np.array(vec)
        else:
            raise AquaError("Vector list must be depth 2 or 3")

    def save_to_list(self, mat):
        if not isinstance(mat, np.ndarray):
            return mat
        if mat.dtype == complex:
            return [mat.real.tolist(), mat.imag.tolist()]
        else:
            return mat.tolist()


