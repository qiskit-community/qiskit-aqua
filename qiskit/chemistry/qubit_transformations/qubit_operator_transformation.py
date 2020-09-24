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

"""TODO"""

from abc import ABC, abstractmethod
from typing import Tuple, List

from qiskit.aqua.operators.legacy import WeightedPauliOperator
from qiskit.aqua.algorithms import MinimumEigensolverResult
from qiskit.chemistry.core import MolecularGroundStateResult
from qiskit.chemistry.drivers import BaseDriver


class QubitOperatorTransformation(ABC):
    """TODO"""

    @abstractmethod
    def transform(self, driver: BaseDriver
                  ) -> Tuple[WeightedPauliOperator, List[WeightedPauliOperator]]:
        """TODO"""
        raise NotImplementedError

    @abstractmethod
    def interpret(self, value: float, state: List[float], aux_values: dict) -> MolecularGroundStateResult:
        """TODO"""
        raise NotImplementedError

    # @abstractmethod
    # def interpret(value, aux_values, circuit, params=None):
    # -> GroundStateResult:  # might be fermionic / bosonic
    #    raise NotImplementedError()
