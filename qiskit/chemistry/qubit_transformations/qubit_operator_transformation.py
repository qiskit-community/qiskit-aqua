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
from typing import Tuple, List

from qiskit.aqua.operators.legacy import WeightedPauliOperator
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.results import EigenstateResult


class QubitOperatorTransformation(ABC):
    """Base class for transformation to qubit operators for chemistry problems"""

    @abstractmethod
    def transform(self, driver: BaseDriver
                  ) -> Tuple[WeightedPauliOperator, List[WeightedPauliOperator]]:
        """transforms to qubit operators """
        raise NotImplementedError

    @abstractmethod
    def interpret(self, eigenstate_result: EigenstateResult) -> EigenstateResult:
        """Interprets an EigenstateResult in the context of this transformation.

        Args:
            eigenstate_result: an eigenstate result object.

        Returns:
            An "interpreted" eigenstate result.
        """
        raise NotImplementedError
