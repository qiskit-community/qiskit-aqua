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

"""Base class for transformation to qubit operators for chemistry problems"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union, Callable, Dict, Any

import numpy as np

from qiskit.aqua.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit.aqua.operators import OperatorBase, WeightedPauliOperator
from qiskit.chemistry import FermionicOperator, BosonicOperator
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.results import EigenstateResult


class Transformation(ABC):
    """Base class for transformation to qubit operators for chemistry problems"""

    @abstractmethod
    def transform(self, driver: BaseDriver,
                  aux_operators: Optional[Union[List[FermionicOperator],
                                                List[BosonicOperator]]] = None
                  ) -> Tuple[OperatorBase, List[OperatorBase]]:
        """Transformation from the ``driver`` to a qubit operator.

        Args:
            driver: A driver encoding the molecule information.
            aux_operators: Additional auxiliary operators to evaluate. Must be of type
                ``FermionicOperator`` if the qubit transformation is fermionic and of type
                ``BosonicOperator`` it is bosonic.

        Returns:
            A qubit operator and a dictionary of auxiliary operators.
        """
        raise NotImplementedError

    def get_default_filter_criterion(self) -> Optional[Callable[[Union[List, np.ndarray], float,
                                                                 Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        aqua.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.
        """
        return None

    @abstractmethod
    def interpret(self, raw_result: Union[EigenstateResult, EigensolverResult,
                                          MinimumEigensolverResult]) -> EigenstateResult:
        """Interprets an EigenstateResult in the context of this transformation.

        Args:
            raw_result: an eigenstate result object.

        Returns:
            An "interpreted" eigenstate result.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def commutation_rule(self) -> bool:
        """Getter of the commutation rule"""
        raise NotImplementedError

    @abstractmethod
    def build_hopping_operators(self, excitations: Union[str, List[List[int]]] = 'sd'
                                ) -> Tuple[Dict[str, WeightedPauliOperator],
                                           Dict[str, List[bool]],
                                           Dict[str, List[Any]]]:
        """Builds the product of raising and lowering operators (basic excitation operators)

        Args:
            excitations: The excitations to be included in the eom pseudo-eigenvalue problem.
                If a string ('s', 'd' or 'sd') then all excitations of the given type will be used.
                Otherwise a list of custom excitations can directly be provided.

        Returns:

        """
        raise NotImplementedError
