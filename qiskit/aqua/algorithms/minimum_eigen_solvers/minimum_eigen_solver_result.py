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

"""The Minimum Eigensolver result."""

import warnings
from typing import Dict, Union
import logging
import numpy as np

from qiskit.aqua.algorithms import AlgorithmResult

logger = logging.getLogger(__name__)


class MinimumEigensolverResult(AlgorithmResult):
    """ Minimum Eigensolver Result."""

    @property
    def eigenvalue(self) -> Union[None, float]:
        """ returns eigen value """
        return self.get('eigenvalue')

    @eigenvalue.setter
    def eigenvalue(self, value: float) -> None:
        """ set eigen value """
        self.data['eigenvalue'] = value

    @property
    def eigenstate(self) -> Union[None, np.ndarray]:
        """ return eigen state """
        return self.get('eigenstate')

    @eigenstate.setter
    def eigenstate(self, value: np.ndarray) -> None:
        """ set eigen state """
        self.data['eigenstate'] = value

    @property
    def aux_operator_eigenvalues(self) -> Union[None, np.ndarray]:
        """ return aux operator eigen values """
        return self.get('aux_operator_eigenvalues')

    @aux_operator_eigenvalues.setter
    def aux_operator_eigenvalues(self, value: np.ndarray) -> None:
        """ set aux operator eigen values """
        self.data['aux_operator_eigenvalues'] = value

    @staticmethod
    def from_dict(a_dict: Dict) -> 'MinimumEigensolverResult':
        """ create new object from a dictionary """
        return MinimumEigensolverResult(a_dict)

    def __getitem__(self, key: object) -> object:
        if key == 'energy':
            warnings.warn('energy deprecated, use eigenvalue property.', DeprecationWarning)
            value = super().__getitem__('eigenvalue')
            return None if value is None else value.real
        elif key == 'energies':
            warnings.warn('energies deprecated, use eigenvalue property.', DeprecationWarning)
            value = super().__getitem__('eigenvalue')
            return None if value is None else [value.real]
        elif key == 'eigvals':
            warnings.warn('eigvals deprecated, use eigenvalue property.', DeprecationWarning)
            value = super().__getitem__('eigenvalue')
            return None if value is None else [value]
        elif key == 'eigvecs':
            warnings.warn('eigvecs deprecated, use eigenstate property.', DeprecationWarning)
            value = super().__getitem__('eigenstate')
            return None if value is None else [value]
        elif key == 'aux_ops':
            warnings.warn('aux_ops deprecated, use aux_operator_eigenvalues property.',
                          DeprecationWarning)
            value = super().__getitem__('aux_operator_eigenvalues')
            return None if value is None else [value]

        return super().__getitem__(key)
