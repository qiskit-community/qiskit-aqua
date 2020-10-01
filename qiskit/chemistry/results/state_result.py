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

"""State results module."""

from typing import Optional
import numpy as np

from qiskit.aqua.algorithms import AlgorithmResult


class StateResult(AlgorithmResult):
    """The state result interface."""

    @property
    def aux_values(self) -> Optional[np.ndarray]:
        """ return aux operator eigen values """
        return self.get('aux_values')

    @aux_values.setter
    def aux_values(self, value: np.ndarray) -> None:
        """ set aux operator eigen values """
        self.data['aux_values'] = value

    @property
    def raw_result(self) -> Optional[AlgorithmResult]:
        """Returns the raw algorithm result."""
        return self.get('raw_result')

    @raw_result.setter
    def raw_result(self, result: AlgorithmResult) -> None:
        self.data['raw_result'] = result


class GroundStateResult(StateResult):
    """The ground state result interface."""
