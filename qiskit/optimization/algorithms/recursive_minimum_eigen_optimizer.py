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

"""A recursive minimal eigen optimizer in Qiskit's optimization module."""

import logging
from copy import deepcopy
from enum import Enum
from typing import Optional, Union, List, Tuple, Dict, cast

import numpy as np
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.aqua.utils.validation import validate_min

from .minimum_eigen_optimizer import MinimumEigenOptimizer, MinimumEigenOptimizationResult
from .optimization_algorithm import (OptimizationResultStatus, OptimizationAlgorithm,
                                     OptimizationResult)
from ..converters.quadratic_program_to_qubo import QuadraticProgramToQubo, QuadraticProgramConverter
from ..exceptions import QiskitOptimizationError
from ..problems import Variable
from ..problems.quadratic_program import QuadraticProgram

logger = logging.getLogger(__name__)


class IntermediateResult(Enum):
    """
    Defines whether the intermediate results of
    :class:`~qiskit.optimization.algorithms.RecursiveMinimumEigenOptimizer`
    at each iteration should be stored and returned to the end user.
    """

    NO_ITERATIONS = 0
    """No intermediate results are stored."""

    LAST_ITERATION = 1
    """Only results from the last iteration are stored."""

    ALL_ITERATIONS = 2
    """All intermediate results are stored."""


class RecursiveMinimumEigenOptimizationResult(OptimizationResult):
    """Recursive Eigen Optimizer Result."""

    def __init__(self, x: Union[List[float], np.ndarray], fval: float, variables: List[Variable],
                 status: OptimizationResultStatus, replacements: Dict[str, Tuple[str, int]],
                 history: Tuple[List[MinimumEigenOptimizationResult], OptimizationResult]) -> None:
        """
        Constructs an instance of the result class.

        Args:
            x: the optimal value found in the optimization.
            fval: the optimal function value.
            variables: the list of variables of the optimization problem.
            status: the termination status of the optimization algorithm.
            replacements: a dictionary of substituted variables. Key is a variable being
                substituted, value is a tuple of substituting variable and a weight, either 1 or -1.
            history: a tuple containing intermediate results. The first element is a list of
                :class:`~qiskit.optimization.algorithms.MinimumEigenOptimizerResult` obtained by
                invoking :class:`~qiskit.optimization.algorithms.MinimumEigenOptimizer` iteratively,
                the second element is an instance of
                :class:`~qiskit.optimization.algorithm.OptimizationResult` obtained at the last step
                via `min_num_vars_optimizer`.
        """
        super().__init__(x, fval, variables, status, None)
        self._replacements = replacements
        self._history = history

    @property
    def replacements(self) -> Dict[str, Tuple[str, int]]:
        """
        Returns a dictionary of substituted variables. Key is a variable being substituted,  value
        is a tuple of substituting variable and a weight, either 1 or -1."""
        return self._replacements

    @property
    def history(self) -> Tuple[List[MinimumEigenOptimizationResult], OptimizationResult]:
        """
        Returns intermediate results. The first element is a list of
        :class:`~qiskit.optimization.algorithms.MinimumEigenOptimizerResult` obtained by invoking
        :class:`~qiskit.optimization.algorithms.MinimumEigenOptimizer` iteratively, the second
        element is an instance of :class:`~qiskit.optimization.algorithm.OptimizationResult`
        obtained at the last step via `min_num_vars_optimizer`.
        """
        return self._history


class RecursiveMinimumEigenOptimizer(OptimizationAlgorithm):
    """A meta-algorithm that applies a recursive optimization.

    The recursive minimum eigen optimizer applies a recursive optimization on top of
    :class:`~qiskit.optimization.algorithms.MinimumEigenOptimizer`.
    The algorithm is introduced in [1].

    Examples:
        Outline of how to use this class:

    .. code-block::

        from qiskit.aqua.algorithms import QAOA
        from qiskit.optimization.problems import QuadraticProgram
        from qiskit.optimization.algorithms import RecursiveMinimumEigenOptimizer
        problem = QuadraticProgram()
        # specify problem here
        # specify minimum eigen solver to be used, e.g., QAOA
        qaoa = QAOA(...)
        optimizer = RecursiveMinimumEigenOptimizer(qaoa)
        result = optimizer.solve(problem)

    References:
        [1]: Bravyi et al. (2019), Obstacles to State Preparation and Variational Optimization
            from Symmetry Protection. http://arxiv.org/abs/1910.08980.
    """

    def __init__(self, min_eigen_optimizer: MinimumEigenOptimizer, min_num_vars: int = 1,
                 min_num_vars_optimizer: Optional[OptimizationAlgorithm] = None,
                 penalty: Optional[float] = None,
                 history: Optional[IntermediateResult] = IntermediateResult.LAST_ITERATION,
                 converters: Optional[Union[QuadraticProgramConverter,
                                            List[QuadraticProgramConverter]]] = None) -> None:
        """ Initializes the recursive minimum eigen optimizer.

        This initializer takes a ``MinimumEigenOptimizer``, the parameters to specify until when to
        to apply the iterative scheme, and the optimizer to be applied once the threshold number of
        variables is reached.

        Args:
            min_eigen_optimizer: The eigen optimizer to use in every iteration.
            min_num_vars: The minimum number of variables to apply the recursive scheme. If this
                threshold is reached, the min_num_vars_optimizer is used.
            min_num_vars_optimizer: This optimizer is used after the recursive scheme for the
                problem with the remaining variables.
            penalty: The factor that is used to scale the penalty terms corresponding to linear
                equality constraints.
            history: Whether the intermediate results are stored.
                Default value is :py:obj:`~IntermediateResult.LAST_ITERATION`.
            converters: The converters to use for converting a problem into a different form.
                By default, when None is specified, an internally created instance of
                :class:`~qiskit.optimization.converters.QuadraticProgramToQubo` will be used.

        Raises:
            QiskitOptimizationError: In case of invalid parameters (num_min_vars < 1).
            TypeError: When there one of converters is an invalid type.
        """

        validate_min('min_num_vars', min_num_vars, 1)

        self._min_eigen_optimizer = min_eigen_optimizer
        self._min_num_vars = min_num_vars
        if min_num_vars_optimizer:
            self._min_num_vars_optimizer = min_num_vars_optimizer
        else:
            self._min_num_vars_optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        self._penalty = penalty
        self._history = history

        self._converters = self._prepare_converters(converters, penalty)

    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem can be converted
        to a QUBO, and otherwise, returns a message explaining the incompatibility.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            A message describing the incompatibility.
        """
        return QuadraticProgramToQubo.get_compatibility_msg(problem)

    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """Tries to solve the given problem using the recursive optimizer.

        Runs the optimizer to try to solve the optimization problem.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: Incompatible problem.
            QiskitOptimizationError: Infeasible due to variable substitution
        """
        self._verify_compatibility(problem)

        # convert problem to QUBO, this implicitly checks if the problem is compatible
        problem_ = self._convert(problem, self._converters)
        problem_ref = deepcopy(problem_)

        # run recursive optimization until the resulting problem is small enough
        replacements = {}  # type: Dict[str, Tuple[str, int]]
        min_eigen_results = []  # type: List[MinimumEigenOptimizationResult]
        while problem_.get_num_vars() > self._min_num_vars:

            # solve current problem with optimizer
            res = self._min_eigen_optimizer.solve(problem_)  # type: MinimumEigenOptimizationResult
            if self._history == IntermediateResult.ALL_ITERATIONS:
                min_eigen_results.append(res)

            # analyze results to get strongest correlation
            correlations = res.get_correlations()
            i, j = self._find_strongest_correlation(correlations)

            x_i = problem_.variables[i].name
            x_j = problem_.variables[j].name
            if correlations[i, j] > 0:
                # set x_i = x_j
                problem_ = problem_.substitute_variables(variables={i: (j, 1)})
                if problem_.status == QuadraticProgram.Status.INFEASIBLE:
                    raise QiskitOptimizationError('Infeasible due to variable substitution')
                replacements[x_i] = (x_j, 1)
            else:
                # set x_i = 1 - x_j, this is done in two steps:
                # 1. set x_i = 1 + x_i
                # 2. set x_i = -x_j

                # 1a. get additional offset
                constant = problem_.objective.constant
                constant += problem_.objective.linear[i]
                constant += problem_.objective.quadratic[i, i]
                problem_.objective.constant = constant

                # 1b. get additional linear part
                for k in range(problem_.get_num_vars()):
                    coeff = problem_.objective.linear[k]
                    if k == i:
                        coeff += 2 * problem_.objective.quadratic[i, k]
                    else:
                        coeff += problem_.objective.quadratic[i, k]

                    # set new coefficient if not too small
                    if np.abs(coeff) > 1e-10:
                        problem_.objective.linear[k] = coeff
                    else:
                        problem_.objective.linear[k] = 0

                # 2. replace x_i by -x_j
                problem_ = problem_.substitute_variables(variables={i: (j, -1)})
                if problem_.status == QuadraticProgram.Status.INFEASIBLE:
                    raise QiskitOptimizationError('Infeasible due to variable substitution')
                replacements[x_i] = (x_j, -1)

        # solve remaining problem
        result = self._min_num_vars_optimizer.solve(problem_)

        # unroll replacements
        var_values = {}
        for i, x in enumerate(problem_.variables):
            var_values[x.name] = result.x[i]

        def find_value(x, replacements, var_values):
            if x in var_values:
                # if value for variable is known, return it
                return var_values[x]
            elif x in replacements:
                # get replacement for variable
                (y, sgn) = replacements[x]
                # find details for replacing variable
                value = find_value(y, replacements, var_values)
                # construct, set, and return new value
                var_values[x] = value if sgn == 1 else 1 - value
                return var_values[x]
            else:
                raise QiskitOptimizationError('Invalid values!')

        # loop over all variables to set their values
        for x_i in problem_ref.variables:
            if x_i.name not in var_values:
                find_value(x_i.name, replacements, var_values)

        # build history before any translations are applied
        # min_eigen_results is an empty list if history is set to NO or LAST.
        history = (min_eigen_results,
                   None if self._history == IntermediateResult.NO_ITERATIONS else result)

        # construct result
        x_v = np.array([var_values[x_aux.name] for x_aux in problem_ref.variables])
        return cast(RecursiveMinimumEigenOptimizationResult,
                    self._interpret(x=x_v, converters=self._converters, problem=problem,
                                    result_class=RecursiveMinimumEigenOptimizationResult,
                                    replacements=replacements, history=history))

    @staticmethod
    def _find_strongest_correlation(correlations):

        # get absolute values and set diagonal to -1 to make sure maximum is always on off-diagonal
        abs_correlations = np.abs(correlations)
        for i in range(len(correlations)):
            abs_correlations[i, i] = -1

        # get index of maximum (by construction on off-diagonal)
        m_max = np.argmax(abs_correlations.flatten())

        # translate back to indices
        i = int(m_max // len(correlations))
        j = int(m_max - i * len(correlations))
        return (i, j)
