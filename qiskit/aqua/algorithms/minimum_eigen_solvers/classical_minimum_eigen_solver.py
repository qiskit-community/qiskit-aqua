# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Minimum Eigensolver algorithm."""

from typing import List, Optional
import logging
import pprint
import numpy as np
from qiskit.aqua.algorithms import ClassicalEigensolver
from qiskit.aqua.operators import BaseOperator
from .minimum_eigen_solver_result import MinimumEigensolverResult

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name

class ClassicalMinimumEigensolver(ClassicalEigensolver):
    """
    The Minimum Eigensolver algorithm.
    """

    def __init__(self, operator: BaseOperator,
                 aux_operators: Optional[List[BaseOperator]] = None) -> None:
        """
        Args:
            operator: Operator instance
            aux_operators: Auxiliary operators to be evaluated at each eigenvalue
        """
        super().__init__(operator, 1, aux_operators)

    def _run(self):
        """
        Run the algorithm to compute up to the requested k number of eigenvalues.
        Returns:
            dict: Dictionary of results
        """
        super()._run()

        logger.debug('ClassicalMinimumEigensolver _run result:\n%s',
                     pprint.pformat(self._ret, indent=4))
        result = MinimumEigensolverResult()
        if 'eigvals' in self._ret and \
                isinstance(self._ret['eigvals'], np.ndarray) and \
                self._ret['eigvals'].size > 0:
            result.eigenvalue = self._ret['eigvals'][0]
        if 'eigvecs' in self._ret and \
                isinstance(self._ret['eigvecs'], np.ndarray) and \
                self._ret['eigvecs'].size > 0:
            result.eigenstate = self._ret['eigvecs'][0]
        if 'aux_ops' in self._ret and \
                isinstance(self._ret['aux_ops'], np.ndarray) and \
                self._ret['aux_ops'].size > 0:
            result.aux_operator_eigenvalues = self._ret['aux_ops'][0]

        logger.debug('MinimumEigensolverResult dict:\n%s',
                     pprint.pformat(result.data, indent=4))

        return result
