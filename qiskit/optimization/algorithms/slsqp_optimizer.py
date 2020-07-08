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

"""The SLSQP optimizer wrapped to be used within Qiskit's optimization module."""
import logging
from typing import List, cast

import numpy as np
from scipy.optimize import fmin_slsqp

from .multistart_optimizer import MultiStartOptimizer
from .optimization_algorithm import OptimizationResult
from ..exceptions import QiskitOptimizationError
from ..problems.constraint import Constraint
from ..problems.quadratic_program import QuadraticProgram

logger = logging.getLogger(__name__)


class SlsqpOptimizer(MultiStartOptimizer):
    """The SciPy SLSQP optimizer wrapped as an Qiskit :class:`OptimizationAlgorithm`.

    This class provides a wrapper for ``scipy.optimize.fmin_slsqp``
    (https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.optimize.fmin_slsqp.html)
    to be used within the optimization module.
    The arguments for ``fmin_slsqp`` are passed via the constructor.

    Examples:
        >>> from qiskit.optimization.problems import QuadraticProgram
        >>> from qiskit.optimization.algorithms import SlsqpOptimizer
        >>> problem = QuadraticProgram()
        >>> # specify problem here
        >>> x = problem.continuous_var(name="x")
        >>> y = problem.continuous_var(name="y")
        >>> problem.maximize(linear=[2, 0], quadratic=[[-1, 2], [0, -2]])
        >>> optimizer = SlsqpOptimizer()
        >>> result = optimizer.solve(problem)
    """

    # pylint: disable=redefined-builtin
    def __init__(self, iter: int = 100, acc: float = 1.0E-6, iprint: int = 0, trials: int = 1,
                 clip: float = 100.) -> None:
        """Initializes the SlsqpOptimizer.

        This initializer takes the algorithmic parameters of SLSQP and stores them for later use
        of ``fmin_slsqp`` when :meth:`solve` is invoked.
        This optimizer can be applied to find a (local) optimum for problems consisting of only
        continuous variables.

        Args:
            iter: The maximum number of iterations.
            acc: Requested accuracy.
            iprint: The verbosity of fmin_slsqp :

                - iprint <= 0 : Silent operation
                - iprint == 1 : Print summary upon completion (default)
                - iprint >= 2 : Print status of each iterate and summary

            trials: The number of trials for multi-start method. The first trial is solved with
                the initial guess of zero. If more than one trial is specified then
                initial guesses are uniformly drawn from ``[lowerbound, upperbound]``
                with potential clipping.
            clip: Clipping parameter for the initial guesses in the multi-start method.
                If a variable is unbounded then the lower bound and/or upper bound are replaced
                with the ``-clip`` or ``clip`` values correspondingly for the initial guesses.
        """

        super().__init__(trials, clip)
        self._iter = iter
        self._acc = acc
        self._iprint = iprint
        self._trials = trials
        self._clip = clip

    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem contains only
        continuous variables, and otherwise, returns a message explaining the incompatibility.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns a string describing the incompatibility.
        """
        # check whether there are variables of type other than continuous
        if len(problem.variables) > problem.get_num_continuous_vars():
            return 'The SLSQP optimizer supports only continuous variables'

        return ''

    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """
        self._verify_compatibility(problem)

        # construct quadratic objective function
        def _objective(x):
            return problem.objective.sense.value * problem.objective.evaluate(x)

        def _objective_gradient(x):
            return problem.objective.sense.value * problem.objective.evaluate_gradient(x)

        # initialize constraints and bounds
        slsqp_bounds = []
        slsqp_eq_constraints = []
        slsqp_ineq_constraints = []

        # add lower/upper bound constraints
        for variable in problem.variables:
            lowerbound = variable.lowerbound
            upperbound = variable.upperbound
            slsqp_bounds.append((lowerbound, upperbound))

        # pylint: disable=no-member
        # add linear and quadratic constraints
        for constraint in cast(List[Constraint], problem.linear_constraints) + \
                cast(List[Constraint], problem.quadratic_constraints):
            rhs = constraint.rhs
            sense = constraint.sense

            if sense == Constraint.Sense.EQ:
                slsqp_eq_constraints += [lambda x, rhs=rhs, c=constraint: rhs - c.evaluate(x)]
            elif sense == Constraint.Sense.LE:
                slsqp_ineq_constraints += [lambda x, rhs=rhs, c=constraint: rhs - c.evaluate(x)]
            elif sense == Constraint.Sense.GE:
                slsqp_ineq_constraints += [lambda x, rhs=rhs, c=constraint: c.evaluate(x) - rhs]
            else:
                raise QiskitOptimizationError('Unsupported constraint type!')

        # actual minimization function to be called by multi_start_solve
        def _minimize(x_0: np.array) -> np.array:
            return fmin_slsqp(_objective, x_0, eqcons=slsqp_eq_constraints,
                              ieqcons=slsqp_ineq_constraints, bounds=slsqp_bounds,
                              fprime=_objective_gradient, iter=self._iter, acc=self._acc,
                              iprint=self._iprint)

        return self.multi_start_solve(_minimize, problem)
