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
"""The Exact Eigensolver algorithm."""

import logging

import numpy as np
from scipy import sparse as scisparse

from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua import AquaError, Pluggable
from qiskit.aqua.operators import MatrixOperator, op_converter  # pylint: disable=unused-import

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class ExactEigensolver(QuantumAlgorithm):
    """The Exact Eigensolver algorithm."""

    CONFIGURATION = {
        'name': 'ExactEigensolver',
        'description': 'ExactEigensolver Algorithm',
        'classical': True,
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
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

    def __init__(self, operator, k=1, aux_operators=None):
        """Constructor.

        Args:
            operator (MatrixOperator): instance
            k (int): How many eigenvalues are to be computed
            aux_operators (list[MatrixOperator]): Auxiliary operators
                        to be evaluated at each eigenvalue
        """
        self.validate(locals())
        super().__init__()

        self._operator = op_converter.to_matrix_operator(operator)
        if aux_operators is None:
            self._aux_operators = []
        else:
            aux_operators = \
                [aux_operators] if not isinstance(aux_operators, list) else aux_operators
            self._aux_operators = \
                [op_converter.to_matrix_operator(aux_op) for aux_op in aux_operators]
        self._k = k
        if self._k > self._operator.matrix.shape[0]:
            self._k = self._operator.matrix.shape[0]
            logger.debug("WARNING: Asked for %s eigenvalues but max possible is %s.", k, self._k)
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params (dict): parameters dictionary
            algo_input (EnergyInput): instance
        Returns:
            ExactEigensolver: an instance of this class
        Raises:
            AquaError: invalid input
        """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")
        ee_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        k = ee_params.get('k')
        return cls(algo_input.qubit_op, k, algo_input.aux_ops)

    def _solve(self):
        if self._operator.dia_matrix is None:
            if self._k >= self._operator.matrix.shape[0] - 1:
                logger.debug("Scipy doesn't support to get all eigenvalues, using numpy instead.")
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
        if self._aux_operators:
            aux_op_vals = np.empty([self._k, len(self._aux_operators), 2])
            for i in range(self._k):
                aux_op_vals[i, :] = self._eval_aux_operators(self._ret['eigvecs'][i])
            self._ret['aux_ops'] = aux_op_vals

    def _eval_aux_operators(self, wavefn, threshold=1e-12):
        values = []
        for operator in self._aux_operators:
            value = 0.0
            if not operator.is_empty():
                value, _ = operator.evaluate_with_statevector(wavefn)
                value = value.real if abs(value.real) > threshold else 0.0
            values.append((value, 0))
        return np.asarray(values)

    def _run(self):
        """
        Run the algorithm to compute up to the requested k number of eigenvalues.
        Returns:
            dict: Dictionary of results
        """
        self._solve()
        self._get_ground_state_energy()
        self._get_energies()
        return self._ret
