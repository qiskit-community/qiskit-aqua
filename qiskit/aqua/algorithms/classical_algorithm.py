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

"""
This module implements the abstract base class for classical algorithm modules.

To create add-on classical algorithm modules subclass the ClassicalAlgorithm
class in this module.
Doing so requires that the required algorithm interface is implemented.
"""

from abc import ABC, abstractmethod
from typing import Dict
from qiskit.aqua import aqua_globals


class ClassicalAlgorithm(ABC):
    """
    Base class for Classical Algorithms.
    """

    @property
    def random(self):
        """Return a numpy random."""
        return aqua_globals.random

    def run(self) -> Dict:
        """Execute the classical algorithm.

        Returns:
            dict: results of an algorithm.
        """

        return self._run()

    @abstractmethod
    def _run(self) -> Dict:
        raise NotImplementedError()
