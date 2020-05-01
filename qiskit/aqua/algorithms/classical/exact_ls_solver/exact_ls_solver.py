# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""The Exact LinearSystem algorithm."""

import logging

import numpy as np

from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua import AquaError

logger = logging.getLogger(__name__)


class ExactLSsolver(QuantumAlgorithm):
    """The Exact LinearSystem algorithm."""

    CONFIGURATION = {
        'name': 'ExactLSsolver',
        'description': 'ExactLSsolver Algorithm',
        'classical': True,
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'ExactLSsolver_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'problems': ['linear_system']
    }

    def __init__(self, matrix=None, vector=None):
        """Constructor.

        Args:
            matrix (array): the input matrix of linear system of equations
            vector (array): the input vector of linear system of equations
        """
        self.validate(locals())
        super().__init__()
        self._matrix = matrix
        self._vector = vector
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):  # pylint: disable=unused-argument
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params (dict): parameters dictionary
            algo_input (LinearSystemInput): instance
        Returns:
            ExactLSsolver: an instance of this class
        Raises:
            AquaError: invalid input
            ValueError: invalid input
        """
        if algo_input is None:
            raise AquaError("LinearSystemInput instance is required.")

        matrix = algo_input.matrix
        vector = algo_input.vector
        if not isinstance(matrix, np.ndarray):
            matrix = np.asarray(matrix)
        if not isinstance(vector, np.ndarray):
            vector = np.asarray(vector)

        if matrix.shape[0] != len(vector):
            raise ValueError("Input vector dimension does not match input "
                             "matrix dimension!")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input matrix must be square!")

        return cls(matrix, vector)

    def _solve(self):
        self._ret['eigvals'] = np.linalg.eig(self._matrix)[0]
        self._ret['solution'] = list(np.linalg.solve(self._matrix, self._vector))

    def _run(self):
        """
        Run the algorithm to compute eigenvalues and solution.
        Returns:
            dict: Dictionary of results
        """
        self._solve()
        return self._ret
