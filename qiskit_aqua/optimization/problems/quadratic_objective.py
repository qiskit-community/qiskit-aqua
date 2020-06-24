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

"""Quadratic Objective."""

from enum import Enum
from typing import Union, List, Dict, Tuple, Any

from numpy import ndarray
from scipy.sparse import spmatrix

from .linear_constraint import LinearExpression
from .quadratic_expression import QuadraticExpression
from .quadratic_program_element import QuadraticProgramElement


class ObjSense(Enum):
    """Objective Sense Type."""
    MINIMIZE = 1
    MAXIMIZE = -1


class QuadraticObjective(QuadraticProgramElement):
    """Representation of quadratic objective function of the form:
    constant + linear * x + x * quadratic * x.
    """

    Sense = ObjSense

    def __init__(self, quadratic_program: Any,
                 constant: float = 0.0,
                 linear: Union[ndarray, spmatrix, List[float], Dict[Union[str, int], float]] = None,
                 quadratic: Union[ndarray, spmatrix, List[List[float]],
                                  Dict[Tuple[Union[int, str], Union[int, str]], float]] = None,
                 sense: ObjSense = ObjSense.MINIMIZE
                 ) -> None:
        """Constructs a quadratic objective function.

        Args:
            quadratic_program: The parent quadratic program.
            constant: The constant offset of the objective.
            linear: The coefficients of the linear part of the objective.
            quadratic: The coefficients of the quadratic part of the objective.
            sense: The optimization sense of the objective.
        """
        super().__init__(quadratic_program)
        self._constant = constant
        if linear is None:
            linear = {}
        self._linear = LinearExpression(quadratic_program, linear)
        if quadratic is None:
            quadratic = {}
        self._quadratic = QuadraticExpression(quadratic_program, quadratic)
        self._sense = sense

    @property
    def constant(self) -> float:
        """Returns the constant part of the objective function.

        Returns:
            The constant part of the objective function.
        """
        return self._constant

    @constant.setter
    def constant(self, constant: float) -> None:
        """Sets the constant part of the objective function.

        Args:
            constant: The constant part of the objective function.
        """
        self._constant = constant

    @property
    def linear(self) -> LinearExpression:
        """Returns the linear part of the objective function.

        Returns:
            The linear part of the objective function.
        """
        return self._linear

    @linear.setter
    def linear(self, linear: Union[ndarray, spmatrix, List[float], Dict[Union[str, int], float]]
               ) -> None:
        """Sets the coefficients of the linear part of the objective function.

        Args:
            linear: The coefficients of the linear part of the objective function.

        """
        self._linear = LinearExpression(self.quadratic_program, linear)

    @property
    def quadratic(self) -> QuadraticExpression:
        """Returns the quadratic part of the objective function.

        Returns:
            The quadratic part of the objective function.
        """
        return self._quadratic

    @quadratic.setter
    def quadratic(self, quadratic: Union[ndarray, spmatrix, List[List[float]],
                                         Dict[Tuple[Union[int, str], Union[int, str]], float]]
                  ) -> None:
        """Sets the coefficients of the quadratic part of the objective function.

        Args:
            quadratic: The coefficients of the quadratic part of the objective function.

        """
        self._quadratic = QuadraticExpression(self.quadratic_program, quadratic)

    @property
    def sense(self) -> ObjSense:
        """Returns the sense of the objective function.

        Returns:
            The sense of the objective function.
        """
        return self._sense

    @sense.setter
    def sense(self, sense: ObjSense) -> None:
        """Sets the sense of the objective function.

        Args:
            sense: The sense of the objective function.
        """
        self._sense = sense

    def evaluate(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> float:
        """Evaluate the quadratic objective for given variable values.

        Args:
            x: The values of the variables to be evaluated.

        Returns:
            The value of the quadratic objective given the variable values.
        """
        return self.constant + self.linear.evaluate(x) + self.quadratic.evaluate(x)

    def evaluate_gradient(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> ndarray:
        """Evaluate the gradient of the quadratic objective for given variable values.

        Args:
            x: The values of the variables to be evaluated.

        Returns:
            The value of the gradient of the quadratic objective given the variable values.
        """
        return self.linear.evaluate_gradient(x) + self.quadratic.evaluate_gradient(x)
