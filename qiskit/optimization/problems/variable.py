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

"""Variable interface"""

from enum import Enum
from typing import Tuple, Union, Any

from .quadratic_program_element import QuadraticProgramElement
from ..exceptions import QiskitOptimizationError
from ..infinity import INFINITY


class VarType(Enum):
    """Constants defining variable type."""
    CONTINUOUS = 0
    BINARY = 1
    INTEGER = 2


class Variable(QuadraticProgramElement):
    """Representation of a variable."""

    Type = VarType

    def __init__(self, quadratic_program: Any, name: str,
                 lowerbound: Union[float, int] = 0,
                 upperbound: Union[float, int] = INFINITY,
                 vartype: VarType = VarType.CONTINUOUS) -> None:
        """Creates a new Variable.

        The variables is exposed by the top-level `QuadraticProgram` class
        in `QuadraticProgram.variables`.  This constructor is not meant to be used
        externally.

        Args:
            quadratic_program: The parent QuadraticProgram.
            name: The variable name.
            lowerbound: The variable lowerbound.
            upperbound: The variable upperbound.
            vartype: The variable type.

        Raises:
            QiskitOptimizationError: if lowerbound is greater than upperbound.
        """
        if lowerbound > upperbound:
            raise QiskitOptimizationError("Lowerbound is greater than upperbound!")

        super().__init__(quadratic_program)
        self._name = name
        self._lowerbound = lowerbound
        self._upperbound = upperbound
        self._vartype = vartype

    @property
    def name(self) -> str:
        """Returns the name of the variable.

        Returns:
            The name of the variable.
        """
        return self._name

    @property
    def lowerbound(self) -> Union[float, int]:
        """Returns the lowerbound of the variable.

        Returns:
            The lower bound of the variable.
        """
        return self._lowerbound

    @lowerbound.setter
    def lowerbound(self, lowerbound: Union[float, int]) -> None:
        """Sets the lowerbound of the variable.

        Args:
            lowerbound: The lower bound of the variable.

        Raises:
            QiskitOptimizationError: if lowerbound is greater than upperbound.
        """
        if lowerbound > self.upperbound:
            raise QiskitOptimizationError("Lowerbound is greater than upperbound!")
        self._lowerbound = lowerbound

    @property
    def upperbound(self) -> Union[float, int]:
        """Returns the upperbound of the variable.

        Returns:
            The upperbound of the variable.
        """
        return self._upperbound

    @upperbound.setter
    def upperbound(self, upperbound: Union[float, int]) -> None:
        """Sets the upperbound of the variable.

        Args:
            upperbound: The upperbound of the variable.

        Raises:
            QiskitOptimizationError: if upperbound is smaller than lowerbound.
        """
        if self.lowerbound > upperbound:
            raise QiskitOptimizationError("Lowerbound is greater than upperbound!")
        self._upperbound = upperbound

    @property
    def vartype(self) -> VarType:
        """Returns the type of the variable.

        Returns:
            The variable type.

        """
        return self._vartype

    @vartype.setter
    def vartype(self, vartype: VarType) -> None:
        """Sets the type of the variable.

        Args:
            vartype: The variable type.
        """
        self._vartype = vartype

    def as_tuple(self) -> Tuple[str, Union[float, int], Union[float, int], VarType]:
        """ Returns a tuple corresponding to this variable.

        Returns:
            A tuple corresponding to this variable consisting of name, lowerbound, upperbound and
            variable type.
        """
        return self.name, self.lowerbound, self.upperbound, self.vartype
