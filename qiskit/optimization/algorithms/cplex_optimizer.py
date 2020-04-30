
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

"""The CPLEX optimizer wrapped to be used within Qiskit Optimization."""

from typing import Optional
import logging

from .optimization_algorithm import OptimizationAlgorithm, OptimizationResult
from ..exceptions import QiskitOptimizationError
from ..problems.quadratic_program import QuadraticProgram

logger = logging.getLogger(__name__)

_HAS_CPLEX = False
try:
    from cplex.exceptions import CplexSolverError
    _HAS_CPLEX = True
except ImportError:
    logger.info('CPLEX is not installed.')


class CplexOptimizer(OptimizationAlgorithm):
    """The CPLEX optimizer wrapped as an Qiskit :class:`OptimizationAlgorithm`.

    This class provides a wrapper for ``cplex.Cplex`` (https://pypi.org/project/cplex/)
    to be used within Qiskit Optimization.

    Examples:
        >>> from qiskit.optimization.problems import QuadraticProgram
        >>> from qiskit.optimization.algorithms import CplexOptimizer
        >>> problem = QuadraticProgram()
        >>> # specify problem here
        >>> optimizer = CplexOptimizer()
        >>> result = optimizer.solve(problem)
    """

    def __init__(self, disp: Optional[bool] = False) -> None:
        """Initializes the CplexOptimizer.

        Args:
            disp: Whether to print CPLEX output or not.

        Raises:
            NameError: CPLEX is not installed.
        """
        if not _HAS_CPLEX:
            raise NameError('CPLEX is not installed.')

        self._disp = disp

    @property
    def disp(self) -> bool:
        """Returns the display setting.

        Returns:
            Whether to print CPLEX information or not.
        """
        return self._disp

    @disp.setter
    def disp(self, disp: bool):
        """Set the display setting.
        Args:
            disp: The display setting.
        """
        self._disp = disp

    # pylint:disable=unused-argument
    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with this optimizer.

        Returns ``''`` since CPLEX accepts all problems that can be modeled using the
        ``QuadraticProgram``. CPLEX may throw an exception in case the problem is determined
        to be non-convex.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            An empty string.
        """
        return ''

    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem. If problem is not convex,
        this optimizer may raise an exception due to incompatibility, depending on the settings.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """

        # convert to CPLEX problem
        cplex = problem.to_docplex().get_cplex()

        # set display setting
        if not self.disp:
            cplex.set_log_stream(None)
            cplex.set_error_stream(None)
            cplex.set_warning_stream(None)
            cplex.set_results_stream(None)

        # solve problem
        try:
            cplex.solve()
        except CplexSolverError:
            raise QiskitOptimizationError('Non convex/symmetric matrix.')

        # process results
        sol = cplex.solution

        # create results
        result = OptimizationResult(sol.get_values(), sol.get_objective_value(), sol)

        # return solution
        return result
