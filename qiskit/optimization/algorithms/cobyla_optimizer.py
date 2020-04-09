
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

"""The COBYLA optimizer wrapped to be used within Qiskit Optimization."""

from typing import Optional

import numpy as np
from scipy.optimize import fmin_cobyla

from qiskit.optimization.algorithms import OptimizationAlgorithm
from qiskit.optimization.results import OptimizationResult
from qiskit.optimization.problems import OptimizationProblem
from qiskit.optimization import QiskitOptimizationError
from qiskit.optimization import infinity


class CobylaOptimizer(OptimizationAlgorithm):
    """The COBYLA optimizer wrapped to be used within Qiskit Optimization.

    This class provides a wrapper for ``scipy.optimize.fmin_cobyla``
    (https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_cobyla.html)
    to be used within Qiskit Optimization.
    The arguments for ``fmin_cobyla`` are passed via the constructor.

    Examples:
        >>> problem = OptimizationProblem()
        >>> # specify problem here
        >>> optimizer = CobylaOptimizer()
        >>> result = optimizer.solve(problem)
    """

    def __init__(self, rhobeg: float = 1.0, rhoend: float = 1e-4, maxfun: int = 1000,
                 disp: Optional[int] = None, catol: float = 2e-4) -> None:
        """Initializes the CobylaOptimizer.

        This initializer takes the algorithmic parameters of COBYLA and stores them for later use
        of ``fmin_cobyla`` when :meth:`solve` is invoked.
        This optimizer can be applied to find a (local) optimum for problems consisting of only
        continuous variables.

        Args:
            rhobeg: Reasonable initial changes to the variables.
            rhoend: Final accuracy in the optimization (not precisely guaranteed).
                This is a lower bound on the size of the trust region.
            disp: Controls the frequency of output; 0 implies no output.
                Feasible values are {0, 1, 2, 3}.
            maxfun: Maximum number of function evaluations.
            catol: Absolute tolerance for constraint violations.
        """

        self._rhobeg = rhobeg
        self._rhoend = rhoend
        self._maxfun = maxfun
        self._disp = disp
        self._catol = catol

    def get_incompatibility(self, problem: OptimizationProblem) -> str:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem contains only
        continuous variables, and otherwise, returns a message explaining the incompatibility.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns a string describing the incompatibility.
        """
        # check whether there are variables of type other than continuous
        if problem.variables.get_num() > problem.variables.get_num_continuous():
            return 'The COBYLA optimizer supports only continuous variables'

        return ''

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """
        # check compatibility and raise exception if incompatible
        msg = self.get_incompatibility(problem)
        if len(msg) > 0:
            raise QiskitOptimizationError('Incompatible problem: {}'.format(msg))

        # get number of variables
        num_vars = problem.variables.get_num()

        # construct objective function from linear and quadratic part of objective
        offset = problem.objective.get_offset()
        linear_dict = problem.objective.get_linear_dict()
        quadratic_dict = problem.objective.get_quadratic_dict()
        linear = np.zeros(num_vars)
        quadratic = np.zeros((num_vars, num_vars))
        for i, v in linear_dict.items():
            linear[i] = v
        for (i, j), v in quadratic_dict.items():
            quadratic[i, j] = v

        def objective(x):
            value = problem.objective.get_sense() * (
                np.dot(linear, x) + np.dot(x, np.dot(quadratic, x)) / 2 + offset
            )
            return value

        # initialize constraints
        constraints = []

        # add variable lower and upper bounds
        lbs = problem.variables.get_lower_bounds()
        ubs = problem.variables.get_upper_bounds()
        # pylint: disable=invalid-sequence-index
        for i in range(num_vars):
            if lbs[i] > -infinity:
                constraints += [lambda x, lbs=lbs, i=i: x - lbs[i]]
            if ubs[i] < infinity:
                constraints += [lambda x, lbs=lbs, i=i: ubs[i] - x]

        # add linear constraints
        for i in range(problem.linear_constraints.get_num()):
            rhs = problem.linear_constraints.get_rhs(i)
            sense = problem.linear_constraints.get_senses(i)
            row = problem.linear_constraints.get_rows(i)
            row_array = np.zeros(num_vars)
            for j, v in zip(row.ind, row.val):
                row_array[j] = v

            if sense == 'E':
                constraints += [
                    lambda x, rhs=rhs, row_array=row_array: rhs - np.dot(x, row_array),
                    lambda x, rhs=rhs, row_array=row_array: np.dot(x, row_array) - rhs
                ]
            elif sense == 'L':
                constraints += [lambda x, rhs=rhs, row_array=row_array: rhs - np.dot(x, row_array)]
            elif sense == 'G':
                constraints += [lambda x, rhs=rhs, row_array=row_array: np.dot(x, row_array) - rhs]
            else:
                # TODO: add range constraints
                raise QiskitOptimizationError('Unsupported constraint type!')

        # add quadratic constraints
        for i in range(problem.quadratic_constraints.get_num()):
            rhs = problem.quadratic_constraints.get_rhs(i)
            sense = problem.quadratic_constraints.get_senses(i)

            linear_comp = problem.quadratic_constraints.get_linear_components(i)
            quadratic_comp = problem.quadratic_constraints.get_quadratic_components(i)

            linear_array = np.zeros(num_vars)
            for j, v in zip(linear_comp.ind, linear_comp.val):
                linear_array[j] = v

            quadratic_array = np.zeros((num_vars, num_vars))
            for j, k, v in zip(quadratic_comp.ind1, quadratic_comp.ind2, quadratic_comp.val):
                quadratic_array[j, k] = v

            def lhs(x, linear_array=linear_array, quadratic_array=quadratic_array):
                return np.dot(x, linear_array) + np.dot(np.dot(x, quadratic_array), x)

            if sense == 'E':
                constraints += [
                    lambda x: rhs - lhs(x),
                    lambda x: lhs(x) - rhs
                ]
            elif sense == 'L':
                constraints += [lambda x: rhs - lhs(x)]
            elif sense == 'G':
                constraints += [lambda x: lhs(x) - rhs]
            else:
                # TODO: add range constraints
                raise QiskitOptimizationError('Unsupported constraint type!')

        # TODO: derive x_0 from lower/upper bounds
        x_0 = np.zeros(problem.variables.get_num())

        # run optimization
        x = fmin_cobyla(objective, x_0, constraints, rhobeg=self._rhobeg, rhoend=self._rhoend,
                        maxfun=self._maxfun, disp=self._disp, catol=self._catol)
        fval = problem.objective.get_sense() * objective(x)

        # return results
        return OptimizationResult(x, fval, x)
