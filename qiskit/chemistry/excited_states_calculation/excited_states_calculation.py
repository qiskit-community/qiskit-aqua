# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" TODO """

from abc import ABC, abstractmethod

from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.ground_state_calculation import GroundStateCalculation


class ExcitedStatesCalculation(ABC):
    """TODO"""

    def __init__(self, ground_state_calculation: GroundStateCalculation) -> None:
        """
        Args:
            ground_state_calculation: TODO
        """
        self._gsc = ground_state_calculation

    @abstractmethod
    def compute_excitedstates(self, driver: BaseDriver):
        """TODO
        Args:
            driver: TODO
        Returns:
            ExcitedStateResult (TODO type hint)
        """
        raise NotImplementedError()