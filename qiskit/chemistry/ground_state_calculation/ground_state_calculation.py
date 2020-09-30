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

"""The ground state calculation interface."""

from abc import ABC, abstractmethod

from qiskit.chemistry import ChemistryResult
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.qubit_transformations import QubitOperatorTransformation


class GroundStateResult(ChemistryResult):
    """The ground state result interface."""


class GroundStateCalculation(ABC):
    """The ground state calculation interface"""

    def __init__(self, transformation: QubitOperatorTransformation) -> None:
        """
        Args:
            transformation: transformation from driver to qubit operator (and aux. operators)
        """
        self._transformation = transformation

    @property
    def transformation(self) -> QubitOperatorTransformation:
        """Returns the transformation used to obtain a qubit operator from the molecule."""
        return self._transformation

    @transformation.setter
    def transformation(self, transformation: QubitOperatorTransformation) -> None:
        """Sets the transformation used to obtain a qubit operator from the molecule."""
        self._transformation = transformation

    @abstractmethod
    def compute_groundstate(self, driver: BaseDriver) -> GroundStateResult:
        """Compute the ground state energy of the molecule that was supplied via the driver.

        Args:
            driver: BaseDriver

        Returns:
            A molecular ground state result
        """
        raise NotImplementedError

    @abstractmethod
    def returns_groundstate(self) -> bool:
        """Whether this class returns only the ground state energy or also the ground state itself.

        Returns:
            True, if this class also returns the ground state in the results object.
            False otherwise.
        """
        raise NotImplementedError
