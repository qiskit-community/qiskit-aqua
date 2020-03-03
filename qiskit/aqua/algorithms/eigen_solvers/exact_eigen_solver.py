# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""The Exact Eigensolver algorithm."""

from typing import List, Optional
import logging

import numpy as np
from scipy import sparse as scisparse

from qiskit.aqua.algorithms import MinEigenSolver
from qiskit.aqua.algorithms import ClassicalAlgorithm
from qiskit.aqua.operators import MatrixOperator, op_converter  # pylint: disable=unused-import
from qiskit.aqua.operators import BaseOperator
from qiskit.aqua.utils.validation import validate_min

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class ExactEigensolver(ClassicalAlgorithm, MinEigenSolver):
    r"""
    The Exact Eigensolver algorithm.

    Exact Eigensolver computes up to the first :math:`k` eigenvalues of a complex-valued square
    matrix of dimension :math:`n \times n`, with :math:`k \leq n`.

    Note:
        Operators are automatically converted to :class:`~qiskit.aqua.operators.MatrixOperator`
        as needed and this conversion can be costly in terms of memory and performance as the
        operator size, mostly in terms of number of qubits it represents, gets larger.
    """

    def __init__(self, operator: Optional[BaseOperator] = None, k: int = 1,
                 aux_operators: Optional[List[BaseOperator]] = None) -> None:
        """
        Args:
            operator: Operator instance
            k: How many eigenvalues are to be computed, has a min. value of 1.
            aux_operators: Auxiliary operators to be evaluated at each eigenvalue
        """
        validate_min('k', k, 1)
        super().__init__()

        if operator:
            self._operator = op_converter.to_matrix_operator(operator)
        else:
            self._operator = None

        if aux_operators is None:
            self._aux_operators = []
        else:
            aux_operators = \
                [aux_operators] if not isinstance(aux_operators, list) else aux_operators
            self._aux_operators = \
                [op_converter.to_matrix_operator(aux_op) for aux_op in aux_operators]
        self._k = k
        if self._operator and self._k > self._operator.matrix.shape[0]:
            self._k = self._operator.matrix.shape[0]
            logger.debug("WARNING: Asked for %s eigenvalues but max possible is %s.", k, self._k)
        self._ret = {}

    @property
    def operator(self):
        """Return the operator."""
        return self._operator

    @operator.setter
    def operator(self, operator):
        self._operator = op_converter.to_matrix_operator(operator)

    def _solve(self):
        if self._operator is None:
            raise ValueError('The operator has not been set!')

        operator = self._operator
        if not isinstance(operator, MatrixOperator):
            operator = op_converter.to_matrix_operator(operator)

        if operator.dia_matrix is None:
            if self._k >= operator.matrix.shape[0] - 1:
                logger.debug("Scipy doesn't support to get all eigenvalues, using numpy instead.")
                eigval, eigvec = np.linalg.eig(operator.matrix.toarray())
            else:
                eigval, eigvec = scisparse.linalg.eigs(operator.matrix, k=self._k, which='SR')
        else:
            eigval = np.sort(operator.matrix.data)[:self._k]
            temp = np.argsort(operator.matrix.data)[:self._k]
            eigvec = np.zeros((operator.matrix.shape[0], self._k))
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

    def compute_min_eigenvalue(self, operator: Optional[BaseOperator] = None
                               ) -> Tuple[List[float], float]:
        # if operator is None, set it to the one given in the initializer
        # if it is still None, raise an error
        operator = operator or self._operator
        if operator is None:
            raise AquaError('Provide an operator either in the initializer or this method.')

        # run the algorithm with the operator passed in
        # (bit hacky w/o the QuantumAlgorithm refactor)
        current_operator = self._operator
        self._operator = operator
        ret = self.run(self._quantum_instance)
        self._operator = current_operator

        # return the eigenvector corresponding to the lowest eigenvalue
        return ret['wavefunction'][0] if self._k == 1 else ret['wavefunction'][0], ret['energy']

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
