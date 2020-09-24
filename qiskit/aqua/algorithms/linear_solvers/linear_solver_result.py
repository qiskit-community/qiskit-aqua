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

"""The Linear solver result."""

from typing import Dict
import numpy as np

from qiskit.aqua.algorithms import AlgorithmResult


class LinearsolverResult(AlgorithmResult):
    """ Linear solver Result."""

    @property
    def solution(self) -> np.ndarray:
        """ return solution """
        return self.get('solution')

    @solution.setter
    def solution(self, value: np.ndarray) -> None:
        """ set solution """
        self.data['solution'] = value

    @staticmethod
    def from_dict(a_dict: Dict) -> 'LinearsolverResult':
        """ create new object from a dictionary """
        return LinearsolverResult(a_dict)
