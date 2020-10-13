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
This module implements the abstract base class for driver modules.
"""

from typing import Optional
from abc import ABC, abstractmethod

from .molecule import Molecule
from ..qiskit_chemistry_error import QiskitChemistryError


class BaseDriver(ABC):
    """
    Base class for Qiskit's chemistry drivers.
    """

    @abstractmethod
    def __init__(self,
                 molecule: Optional[Molecule] = None,
                 basis: str = 'sto3g',
                 hf_method: str = 'rhf',
                 supports_molecule: bool = False) -> None:
        """
        Args:
            molecule: molecule
            basis: basis set
            hf_method: Hartree-Fock Method type
            supports_molecule: Indicates if driver supports molecule

        Raises:
            QiskitChemistryError: Molecule passed but driver doesn't support it.
        """
        if molecule is not None and not supports_molecule:
            raise QiskitChemistryError("Driver doesn't support molecule.")

        self._molecule = molecule
        self._basis = basis
        self._hf_method = hf_method
        self._supports_molecule = supports_molecule

    @property
    def supports_molecule(self) -> bool:
        """
        True for derived classes that support Molecule.

        Returns:
            True if Molecule is supported.
        """
        return self._supports_molecule

    @property
    def molecule(self) -> Optional[Molecule]:
        """ return molecule """
        return self._molecule

    @molecule.setter
    def molecule(self, value: Molecule) -> None:
        """ set molecule """
        if not self.supports_molecule:
            raise QiskitChemistryError("Driver doesn't support molecule.")
        self._molecule = value

    @property
    def basis(self) -> str:
        """ return basis """
        return self._basis

    @basis.setter
    def basis(self, value: str) -> None:
        """ set basis """
        self._basis = value

    @property
    def hf_method(self) -> str:
        """ return Hartree-Fock method """
        return self._hf_method

    @hf_method.setter
    def hf_method(self, value: str) -> None:
        """ set Hartree-Fock method """
        self._hf_method = value
