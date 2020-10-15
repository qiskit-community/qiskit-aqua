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

"""BOPES Sampler result"""

import logging
from typing import List, Dict

from qiskit.chemistry.results import EigenstateResult

logger = logging.getLogger(__name__)


class BOPESSamplerResult:
    """The BOPES Sampler result"""

    def __init__(self, points: List[float],
                 energies: List[float],
                 raw_results: Dict[float, EigenstateResult]) -> None:
        """
        Creates an new instance of the result.
        Args:
            points: List of points.
            energies: List of energies.
            raw_results: Raw results obtained from the solver.
        """
        super().__init__()
        self._points = points
        self._energies = energies
        self._raw_results = raw_results

    @property
    def points(self) -> List[float]:
        """ returns list of points."""
        return self._points

    @property
    def energies(self) -> List[float]:
        """ returns list of energies."""
        return self._energies

    @property
    def raw_results(self) -> Dict[float, EigenstateResult]:
        """ returns all results for all points."""
        return self._raw_results

    def point_results(self, point: float) -> EigenstateResult:
        """ returns all results for a specific point."""
        return self.raw_results[point]
