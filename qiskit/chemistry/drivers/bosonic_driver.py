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

"""
This module implements the abstract base class for bosonic driver modules.
"""

from abc import abstractmethod

from ..watson_hamiltonian import WatsonHamiltonian
from .base_driver import BaseDriver


class BosonicDriver(BaseDriver):
    """
    Base class for Qiskit's chemistry bosonic drivers.
    """

    @abstractmethod
    def run(self) -> WatsonHamiltonian:
        """
        Runs driver to produce a WatsonHamiltonian output.

        Returns:
            A WatsonHamiltonian comprising the bosonic data.
        """
        pass
