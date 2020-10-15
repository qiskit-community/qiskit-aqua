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

"""The numpy minimum eigensolver factory for ground state calculation algorithms."""

from typing import Optional, Union, List, Callable
import numpy as np

from qiskit.aqua.algorithms import MinimumEigensolver, NumPyMinimumEigensolver
from ....transformations.transformation import Transformation
from .minimum_eigensolver_factory import MinimumEigensolverFactory


class NumPyMinimumEigensolverFactory(MinimumEigensolverFactory):
    """A factory to construct a NumPyMinimumEigensolver."""

    def __init__(self,
                 filter_criterion: Callable[[Union[List, np.ndarray], float, Optional[List[float]]],
                                            bool] = None,
                 use_default_filter_criterion: bool = False) -> None:
        """
        Args:
            filter_criterion: callable that allows to filter eigenvalues/eigenstates. The minimum
                eigensolver is only searching over feasible states and returns an eigenstate that
                has the smallest eigenvalue among feasible states. The callable has the signature
                `filter(eigenstate, eigenvalue, aux_values)` and must return a boolean to indicate
                whether to consider this value or not. If there is no
                feasible element, the result can even be empty.
            use_default_filter_criterion: whether to use the transformation's default filter
                criterion if ``filter_criterion`` is ``None``.
        """
        self._filter_criterion = filter_criterion
        self._use_default_filter_criterion = use_default_filter_criterion

    @property
    def filter_criterion(self) -> Callable[[Union[List, np.ndarray], float, Optional[List[float]]],
                                           bool]:
        """ returns filter criterion """
        return self._filter_criterion

    @filter_criterion.setter
    def filter_criterion(self, value: Callable[[Union[List, np.ndarray], float,
                                                Optional[List[float]]], bool]) -> None:
        """ sets filter criterion """
        self._filter_criterion = value

    @property
    def use_default_filter_criterion(self) -> bool:
        """ returns whether to use the default filter criterion """
        return self._use_default_filter_criterion

    @use_default_filter_criterion.setter
    def use_default_filter_criterion(self, value: bool) -> None:
        """ sets whether to use the default filter criterion """
        self._use_default_filter_criterion = value

    def get_solver(self, transformation: Transformation) -> MinimumEigensolver:
        """Returns a NumPyMinimumEigensolver which possibly uses the default filter criterion
        provided by the ``transformation``.

        Args:
            transformation: a fermionic qubit operator transformation.

        Returns:
            A NumPyMinimumEigensolver suitable to compute the ground state of the molecule
            transformed by ``transformation``.
        """
        filter_criterion = self._filter_criterion
        if not filter_criterion and self._use_default_filter_criterion:
            filter_criterion = transformation.get_default_filter_criterion()

        npme = NumPyMinimumEigensolver(filter_criterion=filter_criterion)
        return npme

    def supports_aux_operators(self):
        return NumPyMinimumEigensolver.supports_aux_operators()
