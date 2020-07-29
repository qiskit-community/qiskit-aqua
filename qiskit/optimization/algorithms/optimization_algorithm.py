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
from typing import Any, Optional, Union, List, Dict

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
        variables: The list of variables under optimization.
    """

    Status = OptimizationResultStatus

    def __init__(self, x: Optional[Any] = None, fval: Optional[Any] = None,
                 results: Optional[Any] = None,
                 status: OptimizationResultStatus = OptimizationResultStatus.SUCCESS,
                 variables: Optional[List[Variable]] = None) -> None:
        self._x = x if x is not None else []   # pylint: disable=invalid-name
        self._variables = variables if variables is not None else []
        self._fval = fval
        self._results = results
        self._status = status
        self._variable_names = [variable.name for variable in self._variables]
        self._variables_dict = dict(zip(self._variable_names, self._x))

    def __repr__(self):
        self._x = self._x if self._x is not None else []
        return 'optimal variables: [{}]\noptimal function value: {}\nstatus: {}' \
            .format(','.join([str(x_) for x_ in self._x]), self._fval, self._status.name)

    def __getitem__(self, item: Union[int, str]):
        if isinstance(item, int):
            return self._x[item]
        if isinstance(item, str):
            return self._variables_dict[item]
        raise QiskitOptimizationError("Integer or string parameter required, instead "
                                      + type(item).__name__ + " provided.")

    @property
    def variables_dict(self) -> Optional[Dict[str, int]]:
        """Returns the pairs of variable names and values under optimization.

        Returns:
            The pairs of variable names and values under optimization.
        """
        return self._variables_dict

    @property
    def variable_names(self) -> Optional[List[str]]:
        """Returns the list of variable names under optimization.

        Returns:
            The list of variable names under optimization.
        """
        return self._variable_names

    @property
    def x(self) -> Any:
        """Returns the optimal value found in the optimization.

        Returns:
            The optimal value found in the optimization.
        """
        return self._x

    @x.setter  # type: ignore
    def x(self, x: Any) -> None:
        """Set a new optimal value.

        Args:
            x: The new optimal value.
        """
        self._x = x

    @property
    def fval(self) -> Any:
        """Returns the optimal function value.

        Returns:
            The function value corresponding to the optimal value found in the optimization.
        """
        return self._fval

    @fval.setter  # type: ignore
    def fval(self, fval: Any) -> None:
        """Set a new optimal function value.

        Args:
            fval: The new optimal function value.
        """
        self._fval = fval

    @property
    def results(self) -> Any:
        """Return the original results object from the algorithm.

        Currently a dump for any leftovers.

        Returns:
            Additional result information of the optimization algorithm.
        """
        return self._results

    @results.setter  # type: ignore
    def results(self, results: Any) -> None:
        """Set results.

        Args:
            results: The new additional results of the optimization.
        """
        self._results = results

    @property
    def status(self) -> OptimizationResultStatus:
        """Return the termination status of the algorithm.

        Returns:
            The termination status of the algorithm.
        """
        return self._status

    @status.setter  # type: ignore
    def status(self, status: OptimizationResultStatus) -> None:
        """Set a new termination status.

        Args:
            status: The new termination status.
        """
        self._status = status
