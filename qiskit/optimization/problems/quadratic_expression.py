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

"""Quadratic expression interface."""

from typing import List, Union, Dict, Tuple, Any

import numpy as np
from numpy import ndarray
from scipy.sparse import spmatrix, dok_matrix, tril, triu

from .quadratic_program_element import QuadraticProgramElement
from ..exceptions import QiskitOptimizationError


class QuadraticExpression(QuadraticProgramElement):
    """ Representation of a quadratic expression by its coefficients."""

    def __init__(self, quadratic_program: Any,
                 coefficients: Union[ndarray, spmatrix, List[List[float]],
                                     Dict[Tuple[Union[int, str], Union[int, str]], float]]) -> None:
        """Creates a new quadratic expression.

        The quadratic expression can be defined via an array, a list, a sparse matrix, or a
        dictionary that uses variable names or indices as keys and stores the values internally as a
        dok_matrix. We stores values in a compressed way, i.e., values at symmetric positions are
        summed up in the upper triangle. For example, {(0, 1): 1, (1, 0): 2} -> {(0, 1): 3}.

        Args:
            quadratic_program: The parent QuadraticProgram.
            coefficients: The (sparse) representation of the coefficients.

        """
        super().__init__(quadratic_program)
        self.coefficients = coefficients

    def __getitem__(self, key: Tuple[Union[int, str], Union[int, str]]) -> float:
        """Returns the coefficient where i, j can be a variable names or indices.

        Args:
            key: The tuple of indices or names of the variables corresponding to the coefficient.

        Returns:
            The coefficient corresponding to the addressed variables.
        """
        i, j = key
        if isinstance(i, str):
            i = self.quadratic_program.variables_index[i]
        if isinstance(j, str):
            j = self.quadratic_program.variables_index[j]
        return self.coefficients[min(i, j), max(i, j)]

    def __setitem__(self, key: Tuple[Union[int, str], Union[int, str]], value: float) -> None:
        """Sets the coefficient where i, j can be a variable names or indices.

        Args:
            key: The tuple of indices or names of the variables corresponding to the coefficient.
            value: The coefficient corresponding to the addressed variables.
        """
        i, j = key
        if isinstance(i, str):
            i = self.quadratic_program.variables_index[i]
        if isinstance(j, str):
            j = self.quadratic_program.variables_index[j]
        self.coefficients[min(i, j), max(i, j)] = value

    def _coeffs_to_dok_matrix(self,
                              coefficients: Union[ndarray, spmatrix, List[List[float]],
                                                  Dict[Tuple[Union[int, str], Union[int, str]],
                                                       float]]) -> dok_matrix:
        """Maps given coefficients to a dok_matrix.

        Args:
            coefficients: The coefficients to be mapped.

        Returns:
            The given coefficients as a dok_matrix

        Raises:
            QiskitOptimizationError: if coefficients are given in unsupported format.
        """
        if isinstance(coefficients, (list, ndarray, spmatrix)):
            coefficients = dok_matrix(coefficients)
        elif isinstance(coefficients, dict):
            n = self.quadratic_program.get_num_vars()
            coeffs = dok_matrix((n, n))
            for (i, j), value in coefficients.items():
                if isinstance(i, str):
                    i = self.quadratic_program.variables_index[i]
                if isinstance(j, str):
                    j = self.quadratic_program.variables_index[j]
                coeffs[i, j] = value
            coefficients = coeffs
        else:
            raise QiskitOptimizationError(
                "Unsupported format for coefficients: {}".format(coefficients))
        return self._triangle_matrix(coefficients)

    @staticmethod
    def _triangle_matrix(mat: dok_matrix) -> dok_matrix:
        lower = tril(mat, -1, format='dok')
        # `todok` is necessary because subtraction results in other format
        return (mat + lower.transpose() - lower).todok()

    @staticmethod
    def _symmetric_matrix(mat: dok_matrix) -> dok_matrix:
        upper = triu(mat, 1, format='dok') / 2
        # `todok` is necessary because subtraction results in other format
        return (mat + upper.transpose() - upper).todok()

    @property
    def coefficients(self) -> dok_matrix:
        """ Returns the coefficients of the quadratic expression.

        Returns:
            The coefficients of the quadratic expression.
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(self,
                     coefficients: Union[ndarray, spmatrix, List[List[float]],
                                         Dict[Tuple[Union[int, str], Union[int, str]], float]]
                     ) -> None:
        """Sets the coefficients of the quadratic expression.

        Args:
            coefficients: The coefficients of the quadratic expression.
        """
        self._coefficients = self._coeffs_to_dok_matrix(coefficients)

    def to_array(self, symmetric: bool = False) -> ndarray:
        """Returns the coefficients of the quadratic expression as array.

        Args:
            symmetric: Determines whether the output is in a symmetric form or not.

        Returns:
            An array with the coefficients corresponding to the quadratic expression.
        """
        coeffs = self._symmetric_matrix(self._coefficients) if symmetric else self._coefficients
        return coeffs.toarray()

    def to_dict(self, symmetric: bool = False, use_name: bool = False) \
            -> Dict[Union[Tuple[int, int], Tuple[str, str]], float]:
        """Returns the coefficients of the quadratic expression as dictionary, either using tuples
        of variable names or indices as keys.

        Args:
            symmetric: Determines whether the output is in a symmetric form or not.
            use_name: Determines whether to use index or names to refer to variables.

        Returns:
            An dictionary with the coefficients corresponding to the quadratic expression.
        """
        coeffs = self._symmetric_matrix(self._coefficients) if symmetric else self._coefficients
        if use_name:
            return {(self.quadratic_program.variables[i].name,
                     self.quadratic_program.variables[j].name): v
                    for (i, j), v in coeffs.items()}
        else:
            return dict(coeffs.items())

    def evaluate(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> float:
        """Evaluate the quadratic expression for given variables: x * Q * x.

        Args:
            x: The values of the variables to be evaluated.

        Returns:
            The value of the quadratic expression given the variable values.
        """
        x = self._cast_as_array(x)

        # compute x * Q * x for the quadratic expression
        val = x @ self.coefficients @ x

        # return the result
        return val

    def evaluate_gradient(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> ndarray:
        """Evaluate the gradient of the quadratic expression for given variables.

        Args:
            x: The values of the variables to be evaluated.

        Returns:
            The value of the gradient quadratic expression given the variable values.
        """
        x = self._cast_as_array(x)

        # compute (Q' + Q) * x for the quadratic expression
        val = (self.coefficients.transpose() + self.coefficients) @ x

        # return the result
        return val

    def _cast_as_array(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> \
            Union[dok_matrix, np.ndarray]:
        """Converts input to an array if it is a dictionary or list."""
        if isinstance(x, dict):
            x_aux = np.zeros(self.quadratic_program.get_num_vars())
            for i, v in x.items():
                if isinstance(i, str):
                    i = self.quadratic_program.variables_index[i]
                x_aux[i] = v
            x = x_aux
        if isinstance(x, list):
            x = np.array(x)
        return x
