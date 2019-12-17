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

""" Linear System Input """

import numpy as np

from qiskit.aqua import AquaError
from qiskit.aqua.input import AlgorithmInput


class LinearSystemInput(AlgorithmInput):
    """ Linear System Input """
    PROP_KEY_MATRIX = 'matrix'
    PROP_KEY_VECTOR = 'vector'

    CONFIGURATION = {
        'name': 'LinearSystemInput',
        'description': 'Linear System problem input',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
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
        """ returns matrix """
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        """ sets matrix """
        self._matrix = matrix

    @property
    def vector(self):
        """ returns vector """
        return self._vector

    @vector.setter
    def vector(self, vector):
        """ sets vector """
        self._vector = vector

    def validate(self, args_dict):
        """ validate input """
        params = {}
        for key, value in args_dict.items():
            if key == LinearSystemInput.PROP_KEY_MATRIX:
                value = value.save_to_list() if value is not None else {}
            elif key == LinearSystemInput.PROP_KEY_VECTOR:
                value = value.save_to_list() if value is not None else {}

            params[key] = value

        super().validate(params)

    def to_params(self):
        """ to params """
        params = {}
        params[LinearSystemInput.PROP_KEY_MATRIX] = self.save_to_list(self._matrix)
        params[LinearSystemInput.PROP_KEY_VECTOR] = self.save_to_list(self._vector)
        return params

    @classmethod
    def from_params(cls, params):
        """ from params """
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
        """ load matrix from list """

        def depth(x):
            return isinstance(x, list) and max(map(depth, x))+1
        if depth(mat) == 3:
            return np.array(mat[0])+1j*np.array(mat[1])
        elif depth(mat) == 2:
            return np.array(mat)
        else:
            raise AquaError("Matrix list must be depth 2 or 3")

    @staticmethod
    def load_vec_from_list(vec):
        """ load vector from list """

        def depth(x):
            return isinstance(x, list) and max(map(depth, x))+1
        if depth(vec) == 2:
            return np.array(vec[0])+1j*np.array(vec[1])
        elif depth(vec) == 1:
            return np.array(vec)
        else:
            raise AquaError("Vector list must be depth 2 or 3")

    def save_to_list(self, mat):
        """ save to list """
        if not isinstance(mat, np.ndarray):
            return mat
        if mat.dtype == complex:
            return [mat.real.tolist(), mat.imag.tolist()]
        else:
            return mat.tolist()
