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
from typing import Tuple, Dict, Any, Optional

from qiskit.aqua.operators.legacy import WeightedPauliOperator
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.results import StateResult


class QubitOperatorTransformation(ABC):
    """Base class for transformation to qubit operators for chemistry problems"""

    @abstractmethod
    def transform(self, driver: BaseDriver,
                  additional_operators: Optional[Dict[str, Any]] = None
                  ) -> Tuple[WeightedPauliOperator, Dict[str, WeightedPauliOperator]]:
        """transforms to qubit operators """
        raise NotImplementedError

    @abstractmethod
    def add_context(self, result: StateResult) -> None:
        """Adds contextual information to the state result object.

        Args:
            result: a state result object.
        """
        raise NotImplementedError
