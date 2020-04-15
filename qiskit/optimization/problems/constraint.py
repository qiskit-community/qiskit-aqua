# -*- coding: utf-8 -*-

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

"""Abstract Constraint."""

from abc import abstractmethod
from enum import Enum
from typing import Union, List, Dict

from numpy import ndarray

from qiskit.optimization.problems.has_quadratic_program import HasQuadraticProgram


class ConstraintSense(Enum):
    """Constants Sense Type."""
    leq = 0
    geq = 1
    eq = 2  # pylint: disable=locally-disabled, invalid-name


class Constraint(HasQuadraticProgram):
    """Abstract Constraint Class."""

    def __init__(self, quadratic_program: "QuadraticProgram", name: str, sense: ConstraintSense,
                 rhs: float) -> None:
        """ Initializes the constraint.

        Args:
            quadratic_program: The parent QuadraticProgram.
            name: The name of the constraint.
            sense: The sense of the constraint.
            rhs: The right-hand-side of the constraint.
        """
        super().__init__(quadratic_program)
        self._name = name
        self._sense = sense
        self._rhs = rhs

    @property
    def name(self) -> str:
        """Returns the name of the constraint.

        Returns:
            The name of the constraint.
        """
        return self._name

    @property
    def sense(self) -> ConstraintSense:
        """Returns the sense of the constraint.

        Returns:
            The sense of the constraint.
        """
        return self._sense

    @sense.setter
    def sense(self, sense: ConstraintSense) -> None:
        """Sets the sense of the constraint.

        Args:
            sense: The sense of the constraint.
        """
        self._sense = sense

    @property
    def rhs(self) -> float:
        """Returns the right-hand-side of the constraint.

        Returns:
            The right-hand-side of the constraint.
        """
        return self._rhs

    @rhs.setter
    def rhs(self, rhs: ConstraintSense) -> None:
        """Sets the right-hand-side of the constraint.

        Args:
            rhs: The right-hand-side of the constraint.
        """
        self._rhs = rhs

    @abstractmethod
    def evaluate(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> float:
        """Evaluate left-hand-side of constraint for given values of variables.

        Args:
            x: The values to be used for the variables.

        Returns:
            The left-hand-side of the constraint.
        """
        raise NotImplementedError()
