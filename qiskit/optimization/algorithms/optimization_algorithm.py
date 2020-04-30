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
from typing import Any, Optional

from ..problems.quadratic_program import QuadraticProgram


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


class OptimizationResultStatus(Enum):
    """Feasible values for the termination status of an optimization algorithm."""
    SUCCESS = 0
    FAILURE = 1
    INFEASIBLE = 2


class OptimizationResult:
    """The optimization result class.

    The optimization algorithms return an object of the type `OptimizationResult`, which enforces
    providing the following attributes.

    Attributes:
        x: The optimal value found in the optimization algorithm.
        fval: The function value corresponding to the optimal value.
        results: The original results object returned from the optimization algorithm. This can
            contain more information than only the optimal value and function value.
        status: The termination status of the algorithm.
    """

    Status = OptimizationResultStatus

    def __init__(self, x: Optional[Any] = None, fval: Optional[Any] = None,
                 results: Optional[Any] = None,
                 status: OptimizationResultStatus = OptimizationResultStatus.SUCCESS) -> None:
        self._val = x
        self._fval = fval
        self._results = results
        self._status = status

    def __repr__(self):
        return '([{}] / {} / {})'.format(','.join([str(x_) for x_ in self.x]), self.fval,
                                         self.status)

    def __str__(self):
        return 'x=[{}], fval={}'.format(','.join([str(x_) for x_ in self.x]), self.fval)

    @property
    def x(self) -> Any:
        """Returns the optimal value found in the optimization.

        Returns:
            The optimal value found in the optimization.
        """
        return self._val

    @property
    def fval(self) -> Any:
        """Returns the optimal function value.

        Returns:
            The function value corresponding to the optimal value found in the optimization.
        """
        return self._fval

    @property
    def results(self) -> Any:
        """Return the original results object from the algorithm.

        Currently a dump for any leftovers.

        Returns:
            Additional result information of the optimization algorithm.
        """
        return self._results

    @property
    def status(self) -> OptimizationResultStatus:
        """Return the termination status of the algorithm.

        Returns:
            The termination status of the algorithm.
        """
        return self._status

    @x.setter
    def x(self, x: Any) -> None:
        """Set a new optimal value.

        Args:
            x: The new optimal value.
        """
        self._val = x

    @fval.setter
    def fval(self, fval: Any) -> None:
        """Set a new optimal function value.

        Args:
            fval: The new optimal function value.
        """
        self._fval = fval

    @results.setter
    def results(self, results: Any) -> None:
        """Set results.

        Args:
            results: The new additional results of the optimization.
        """
        self._results = results

    @status.setter
    def status(self, status: OptimizationResultStatus) -> None:
        """Set a new termination status.

        Args:
            status: The new termination status.
        """
        self._status = status
