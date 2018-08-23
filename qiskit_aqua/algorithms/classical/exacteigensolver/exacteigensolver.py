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
"""The Exact Eigensolver algorithm."""

import logging

import numpy as np
from scipy import sparse as scisparse
import copy
from qiskit_aqua import QuantumAlgorithm
from qiskit_aqua import AlgorithmError

logger = logging.getLogger(__name__)


class ExactEigensolver(QuantumAlgorithm):
    """The Exact Eigensolver algorithm."""

    PROP_K = 'k'

    EXACTEIGENSOLVER_CONFIGURATION = {
        'name': 'ExactEigensolver',
        'description': 'ExactEigensolver Algorithm',
        'classical': True,
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'ExactEigensolver_schema',
            'type': 'object',
            'properties': {
                'k': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['energy', 'excited_states', 'ising']
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or copy.deepcopy(ExactEigensolver.EXACTEIGENSOLVER_CONFIGURATION))
        self._operator = None
        self._aux_operators = None
        self._k = 1
        self._ret = {}

    def init_params(self, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: EnergyInput instance
        """
        if algo_input is None:
            raise AlgorithmError("EnergyInput instance is required.")
        ee_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        k = ee_params.get(ExactEigensolver.PROP_K)
        self.init_args(algo_input.qubit_op, k, algo_input.aux_ops)

    def init_args(self, operator, k=1, aux_operators=[]):
        """
        Initialize directly via method parameters
        Args:
            operator: Operator instance
            k: How many eigenvalues are to be computed
            aux_operators: Auxiliary operators to be evaluated at each eigenvalue
        """
        self._operator = operator
        if not isinstance(aux_operators, list):
            aux_operators = [aux_operators]
        self._aux_operators = aux_operators
        self._k = k
        self._operator.convert('paulis', 'matrix')
        if self._k > self._operator.matrix.shape[0]:
            self._k = self._operator.matrix.shape[0]
            logger.debug("WARNING: Asked for {} eigenvalues but max possible is {}.".format(k, self._k))
        self._ret = {}

    def _solve(self):
        if self._operator.matrix.ndim == 2:
            if self._k >= self._operator.matrix.shape[0] - 1:
                logger.debug("WARNING: Scipy doesn't support to get all eigenvalues, using numpy instead.")
                eigval, eigvec = np.linalg.eig(self._operator.matrix.toarray())
            else:
                eigval, eigvec = scisparse.linalg.eigs(self._operator.matrix, k=self._k, which='SR')
        else:
            eigval = np.sort(self._operator.matrix.data)[:self._k]
            temp = np.argsort(self._operator.matrix.data)[:self._k]
            eigvec = np.zeros((self._operator.matrix.shape[0], self._k))
            for i, idx in enumerate(temp):
                eigvec[idx, i] = 1.0
        if self._k > 1:
            idx = eigval.argsort()
            eigval = eigval[idx]
            eigvec = eigvec[:, idx]
        self._ret['eigvals'] = eigval
        self._ret['eigvecs'] = eigvec.T

    def _get_ground_state_energy(self):
        if 'eigvals' not in self._ret or 'eigvecs' not in self._ret:
            self._solve()
        self._ret['energy'] = self._ret['eigvals'][0].real
        self._ret['wavefunction'] = self._ret['eigvecs']

    def _get_energies(self):
        if 'eigvals' not in self._ret or 'eigvecs' not in self._ret:
            self._solve()
        energies = np.empty(self._k)
        for i in range(self._k):
            energies[i] = self._ret['eigvals'][i].real
        self._ret['energies'] = energies
        if len(self._aux_operators) > 0:
            aux_op_vals = np.empty([self._k, len(self._aux_operators), 2])
            for i in range(self._k):
                aux_op_vals[i, :] = self._eval_aux_operators(self._ret['eigvecs'][i])
            self._ret['aux_ops'] = aux_op_vals

    def _eval_aux_operators(self, wavefn, threshold=1e-12):
        values = []
        for operator in self._aux_operators:
            operator.convert('paulis', 'matrix')
            value = 0.0
            if not operator.is_empty():
                value, _ = operator.eval('matrix', wavefn, self._backend)
                value = value.real if abs(value.real) > threshold else 0.0
            values.append((value, 0))
        return np.asarray(values)

    def run(self):
        """
        Runs the algorithm to compute up to the requested k number of eigenvalues
        Returns:
            Dictionary of results
        """
        self._solve()
        self._get_ground_state_energy()
        self._get_energies()
        return self._ret
