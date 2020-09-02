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

"""Linear expression interface."""

from typing import List, Union, Dict, Any

from numpy import ndarray
from scipy.sparse import spmatrix, dok_matrix

from .quadratic_program_element import QuadraticProgramElement
from ..exceptions import QiskitOptimizationError


class LinearExpression(QuadraticProgramElement):
    """ Representation of a linear expression by its coefficients."""

    def __init__(self, quadratic_program: Any,
                 coefficients: Union[ndarray, spmatrix, List[float],
                                     Dict[Union[int, str], float]]) -> None:
        """Creates a new linear expression.

        The linear expression can be defined via an array, a list, a sparse matrix, or a dictionary
        that uses variable names or indices as keys and stores the values internally as a
        dok_matrix.

        Args:
            quadratic_program: The parent QuadraticProgram.
            coefficients: The (sparse) representation of the coefficients.

        """
        super().__init__(quadratic_program)
        self.coefficients = coefficients

    def __getitem__(self, i: Union[int, str]) -> float:
        """Returns the i-th coefficient where i can be a variable name or index.

        Args:
            i: the index or name of the variable corresponding to the coefficient.

        Returns:
            The coefficient corresponding to the addressed variable.
        """
        if isinstance(i, str):
            i = self.quadratic_program.variables_index[i]
        return self.coefficients[0, i]

    def __setitem__(self, i: Union[int, str], value: float) -> None:
        if isinstance(i, str):
            i = self.quadratic_program.variables_index[i]
        self._coefficients[0, i] = value

    def _coeffs_to_dok_matrix(self,
                              coefficients: Union[ndarray, spmatrix,
                                                  List, Dict[Union[int, str], float]]
                              ) -> dok_matrix:
        """Maps given 1d-coefficients to a dok_matrix.

        Args:
            coefficients: The 1d-coefficients to be mapped.

        Returns:
            The given 1d-coefficients as a dok_matrix

        Raises:
            QiskitOptimizationError: if coefficients are given in unsupported format.
        """
        if isinstance(coefficients, list) or \
                isinstance(coefficients, ndarray) and len(coefficients.shape) == 1:
            coefficients = dok_matrix([coefficients])
        elif isinstance(coefficients, spmatrix):
            coefficients = dok_matrix(coefficients)
        elif isinstance(coefficients, dict):
            coeffs = dok_matrix((1, self.quadratic_program.get_num_vars()))
            for index, value in coefficients.items():
                if isinstance(index, str):
                    index = self.quadratic_program.variables_index[index]
                coeffs[0, index] = value
            coefficients = coeffs
        else:
            raise QiskitOptimizationError("Unsupported format for coefficients.")
        return coefficients

    @property
    def coefficients(self) -> dok_matrix:
        """ Returns the coefficients of the linear expression.

        Returns:
            The coefficients of the linear expression.
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(self,
                     coefficients: Union[ndarray, spmatrix,
                                         List[float], Dict[Union[str, int], float]]
                     ) -> None:
        """Sets the coefficients of the linear expression.

        Args:
            coefficients: The coefficients of the linear expression.
        """
        self._coefficients = self._coeffs_to_dok_matrix(coefficients)

    def to_array(self) -> ndarray:
        """Returns the coefficients of the linear expression as array.

        Returns:
            An array with the coefficients corresponding to the linear expression.
        """
        return self._coefficients.toarray()[0]

    def to_dict(self, use_name: bool = False) -> Dict[Union[int, str], float]:
        """Returns the coefficients of the linear expression as dictionary, either using variable
        names or indices as keys.

        Args:
            use_name: Determines whether to use index or names to refer to variables.

        Returns:
            An dictionary with the coefficients corresponding to the linear expression.
        """
        if use_name:
            return {self.quadratic_program.variables[k].name: v
                    for (_, k), v in self._coefficients.items()}
        else:
            return {k: v for (_, k), v in self._coefficients.items()}

    def evaluate(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> float:
        """Evaluate the linear expression for given variables.

        Args:
            x: The values of the variables to be evaluated.

        Returns:
            The value of the linear expression given the variable values.
        """
        # cast input to dok_matrix if it is a dictionary
        x = self._coeffs_to_dok_matrix(x)

        # compute the dot-product of the input and the linear coefficients
        val = (x @ self.coefficients.transpose())[0, 0]

        # return the result
        return val

    # pylint: disable=unused-argument
    def evaluate_gradient(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> ndarray:
        """Evaluate the gradient of the linear expression for given variables.

        Args:
            x: The values of the variables to be evaluated.

        Returns:
            The value of the gradient of the linear expression given the variable values.
        """

        # extract the coefficients as array and return it
        return self.to_array()
