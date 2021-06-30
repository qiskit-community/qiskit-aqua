# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The vibronic structure result."""

import logging
from typing import List, Optional

import numpy as np

from qiskit.aqua.algorithms import AlgorithmResult
from .eigenstate_result import EigenstateResult

logger = logging.getLogger(__name__)


class VibronicStructureResult(EigenstateResult):
    """The vibronic structure result."""

    @property
    def algorithm_result(self) -> AlgorithmResult:
        """ Returns raw algorithm result """
        return self.get('algorithm_result')

    @algorithm_result.setter
    def algorithm_result(self, value: AlgorithmResult) -> None:
        """ Sets raw algorithm result """
        self.data['algorithm_result'] = value

    # TODO we need to be able to extract the statevector or the optimal parameters that can
    # construct the circuit of the GS from here (if the algorithm supports this)

    @property
    def computed_vibronic_energies(self) -> np.ndarray:
        """ Returns computed electronic part of ground state energy """
        return self.get('computed_vibronic_energies')

    @computed_vibronic_energies.setter
    def computed_vibronic_energies(self, value: np.ndarray) -> None:
        """ Sets computed electronic part of ground state energy """
        self.data['computed_vibronic_energies'] = value

    @property
    def num_occupied_modals_per_mode(self) -> Optional[List[float]]:
        """ Returns the number of occupied modal per mode """
        return self.get('num_occupied_modals_per_mode')

    @num_occupied_modals_per_mode.setter
    def num_occupied_modals_per_mode(self, value: List[float]) -> None:
        """ Sets measured number of modes """
        self.data['num_occupied_modals_per_mode'] = value

    def __str__(self) -> str:
        """ Printable formatted result """
        return '\n'.join(self.formatted)

    @property
    def formatted(self) -> List[str]:
        """ Formatted result as a list of strings """
        lines = []
        lines.append('=== GROUND STATE ENERGY ===')
        lines.append(' ')
        lines.append('* Vibronic ground state energy (cm^-1): {}'.
                     format(np.round(self.computed_vibronic_energies[0], 12)))
        if len(self.num_occupied_modals_per_mode) > 0:
            lines.append('The number of occupied modals is')
        for i, _ in enumerate(self.num_occupied_modals_per_mode):
            lines.append('- Mode {}: {}'.format(i, self.num_occupied_modals_per_mode[i]))

        return lines
