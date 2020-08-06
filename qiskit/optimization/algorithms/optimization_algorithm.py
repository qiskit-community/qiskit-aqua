# -*- coding: utf-8 -*-

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

"""An abstract class for optimization algorithms in Qiskit's optimization module."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union, Any, Optional, Dict

import numpy as np

from .. import QiskitOptimizationError
from ..problems.quadratic_program import QuadraticProgram, Variable


class OptimizationAlgorithm(ABC):
    """An abstract class for optimization algorithms in Qiskit's optimization module."""

    @abstractmethod
    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with the optimizer implementing this method.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns the incompatibility message. If the message is empty no issues were found.
        """

    def is_compatible(self, problem: QuadraticProgram) -> bool:
        """Checks whether a given problem can be solved with the optimizer implementing this method.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns True if the problem is compatible, False otherwise.
        """
        return len(self.get_compatibility_msg(problem)) == 0

    @abstractmethod
    def solve(self, problem: QuadraticProgram) -> 'OptimizationResult':
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """
        raise NotImplementedError

    def _verify_compatibility(self, problem: QuadraticProgram) -> None:
        """Verifies that the problem is suitable for this optimizer. If the problem is not
        compatible then an exception is raised. This method is for convenience for concrete
        optimizers and is not intended to be used by end user.

        Args:
            problem: Problem to verify.

        Returns:
            None

        Raises:
            QiskitOptimizationError: If the problem is incompatible with the optimizer.

        """
        # check compatibility and raise exception if incompatible
        msg = self.get_compatibility_msg(problem)
        if msg:
            raise QiskitOptimizationError('Incompatible problem: {}'.format(msg))


class OptimizationResultStatus(Enum):
    """Termination status of an optimization algorithm."""

    SUCCESS = 0
    """the optimization algorithm succeeded to find an optimal solution."""

    FAILURE = 1
    """the optimization algorithm ended in a failure."""

    INFEASIBLE = 2
    """the optimization algorithm obtained an infeasible solution."""


class OptimizationResult:
    """The optimization result class.

    The optimization algorithms return an object of the type ``OptimizationResult``
    with the information about the solution obtained.

    ``OptimizationResult`` allows users to get the value of a variable by specifying an index or
    a name as follows.

    Examples:
        >>> from qiskit.optimization import QuadraticProgram
        >>> from qiskit.optimization.algorithms import CplexOptimizer
        >>> problem = QuadraticProgram()
        >>> _ = problem.binary_var('x1')
        >>> _ = problem.binary_var('x2')
        >>> _ = problem.binary_var('x3')
        >>> problem.minimize(linear={'x1': 1, 'x2': -2, 'x3': 3})
        >>> print([var.name for var in problem.variables])
        ['x1', 'x2', 'x3']
        >>> optimizer = CplexOptimizer()
        >>> result = optimizer.solve(problem)
        >>> print(result.variable_names)
        ['x1', 'x2', 'x3']
        >>> print(result.x)
        [0. 1. 0.]
        >>> print(result[1])
        1.0
        >>> print(result['x1'])
        0.0
        >>> print(result.fval)
        -2.0
        >>> print(result.variables_dict)
        {'x1': 0.0, 'x2': 1.0, 'x3': 0.0}

    Note:
        The order of variables should be equal to that of the problem solved by
        optimization algorithms. Optimization algorithms and converters of ``QuadraticProgram``
        should maintain the order when generating a new ``OptimizationResult`` object.
    """

    def __init__(self, x: Union[List[float], np.ndarray], fval: float,
                 variables: List[Variable],
                 raw_results: Optional[Any] = None,
                 status: OptimizationResultStatus = OptimizationResultStatus.SUCCESS) -> None:
        """
        Args:
            x: the optimal value found in the optimization.
            fval: the optimal function value.
            variables: the list of variables of the optimization problem.
            raw_results: the original results object from the optimization algorithm.
            status: the termination status of the optimization algorithm.

        Raises:
            QiskitOptimizationError: if sizes of ``x`` and ``variables`` do not match.
        """
        if len(x) != len(variables):
            raise QiskitOptimizationError(
                'Inconsistent size of optimal value and variables. x: size {} {}, '
                'variables: size {} {}'.format(len(x), x, len(variables),
                                               [v.name for v in variables]))
        self._x = x if isinstance(x, np.ndarray) else np.array(x)  # pylint: disable=invalid-name
        self._fval = fval
        self._raw_results = raw_results
        self._status = status
        self._variables = variables
        self._variable_names = [var.name for var in self._variables]
        self._variables_dict = dict(zip(self._variable_names, self._x))

    def __repr__(self) -> str:
        return 'optimal function value: {}\n' \
               'optimal value: {}\n' \
               'status: {}'.format(self._fval, self._x, self._status.name)

    def __getitem__(self, key: Union[int, str]) -> float:
        """Returns the value of the variable whose index or name is equal to ``key``.

        The key can be an integer or a string.
        If the key is an integer, this methods returns the value of the variable
        whose index is equal to ``key``.
        If the key is a string, this methods return the value of the variable
        whose name is equal to ``key``.

        Args:
            key: an integer or a string.

        Returns:
            The value of a variable whose index or name is equal to ``key``.

        Raises:
            IndexError: if ``key`` is an integer and is out of range of the variables.
            KeyError: if ``key`` is a string and none of the variables has ``key`` as name.
            TypeError: if ``key`` is neither an integer nor a string.
        """
        if isinstance(key, int):
            return self._x[key]
        if isinstance(key, str):
            return self._variables_dict[key]
        raise TypeError(
            "Integer or string key required,"
            "instead {}({}) provided.".format(type(key), key))

    @property
    def x(self) -> np.ndarray:
        """Returns the optimal value found in the optimization.

        Returns:
            The optimal value found in the optimization.
        """
        return self._x

    @property
    def fval(self) -> float:
        """Returns the optimal function value.

        Returns:
            The function value corresponding to the optimal value found in the optimization.
        """
        return self._fval

    @property
    def raw_results(self) -> Any:
        """Return the original results object from the optimization algorithm.

        Currently a dump for any leftovers.

        Returns:
            Additional result information of the optimization algorithm.
        """
        return self._raw_results

    @property
    def status(self) -> OptimizationResultStatus:
        """Returns the termination status of the optimization algorithm.

        Returns:
            The termination status of the algorithm.
        """
        return self._status

    @property
    def variables(self) -> List[Variable]:
        """Returns the list of variables of the optimization problem.

        Returns:
            The list of variables.
        """
        return self._variables

    @property
    def variables_dict(self) -> Dict[str, float]:
        """Returns the optimal value as a dictionary of the variable name and corresponding value.

        Returns:
            The optimal value as a dictionary of the variable name and corresponding value.
        """
        return self._variables_dict

    @property
    def variable_names(self) -> List[str]:
        """Returns the list of variable names of the optimization problem.

        Returns:
            The list of variable names of the optimization problem.
        """
        return self._variable_names
