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

"""Quadratic Constraint."""

from typing import Union, List, Dict, Tuple, Any

from numpy import ndarray
from scipy.sparse import spmatrix

from .constraint import Constraint, ConstraintSense
from .linear_expression import LinearExpression
from .quadratic_expression import QuadraticExpression


class QuadraticConstraint(Constraint):
    """ Representation of a quadratic constraint."""

    # Note: added, duplicating in effect that in Constraint, to avoid issues with Sphinx
    Sense = ConstraintSense

    def __init__(self,
                 quadratic_program: Any, name: str,
                 linear: Union[ndarray, spmatrix, List[float], Dict[Union[str, int], float]],
                 quadratic: Union[ndarray, spmatrix, List[List[float]],
                                  Dict[Tuple[Union[int, str], Union[int, str]], float]],
                 sense: ConstraintSense,
                 rhs: float
                 ) -> None:
        """Constructs a quadratic constraint, consisting of a linear and a quadratic term.

        Args:
            quadratic_program: The parent quadratic program.
            name: The name of the constraint.
            linear: The coefficients specifying the linear part of the constraint.
            quadratic: The coefficients specifying the linear part of the constraint.
            sense: The sense of the constraint.
            rhs: The right-hand-side of the constraint.
        """
        super().__init__(quadratic_program, name, sense, rhs)
        self._linear = LinearExpression(quadratic_program, linear)
        self._quadratic = QuadraticExpression(quadratic_program, quadratic)

    @property
    def linear(self) -> LinearExpression:
        """Returns the linear expression corresponding to the left-hand-side of the constraint.

        Returns:
            The left-hand-side linear expression.
        """
        return self._linear

    @linear.setter
    def linear(self, linear: Union[ndarray, spmatrix, List[float],
                                   Dict[Union[str, int], float]]) -> None:
        """Sets the linear expression corresponding to the left-hand-side of the constraint.
        The coefficients can either be given by an array, a (sparse) 1d matrix, a list or a
        dictionary.

        Args:
            linear: The linear coefficients of the left-hand-side.
        """

        self._linear = LinearExpression(self.quadratic_program, linear)

    @property
    def quadratic(self) -> QuadraticExpression:
        """Returns the quadratic expression corresponding to the left-hand-side of the constraint.

        Returns:
            The left-hand-side quadratic expression.
        """
        return self._quadratic

    @quadratic.setter
    def quadratic(self, quadratic: Union[ndarray, spmatrix, List[List[float]],
                                         Dict[Tuple[Union[int, str], Union[int, str]], float]]) \
            -> None:
        """Sets the quadratic expression corresponding to the left-hand-side of the constraint.
        The coefficients can either be given by an array, a (sparse) matrix, a list or a
        dictionary.

        Args:
            quadratic: The quadratic coefficients of the left-hand-side.
        """
        self._quadratic = QuadraticExpression(self.quadratic_program, quadratic)

    def evaluate(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> float:
        """Evaluate the left-hand-side of the constraint.

        Args:
            x: The values of the variables to be evaluated.

        Returns:
            The left-hand-side of the constraint given the variable values.
        """
        return self.linear.evaluate(x) + self.quadratic.evaluate(x)
