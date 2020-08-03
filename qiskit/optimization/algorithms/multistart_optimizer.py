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

"""Defines an abstract class for multi start optimizers. A multi start optimizer is an optimizer
that may run minimization algorithm for the several time with different initial guesses to achieve
better results. This implementation is suitable for local optimizers."""

import logging
import time
from abc import ABC
from typing import Optional, Callable

import numpy as np
from scipy.stats import uniform

from qiskit.optimization import QuadraticProgram, INFINITY
from qiskit.optimization.algorithms import OptimizationAlgorithm, OptimizationResult

logger = logging.getLogger(__name__)


# we disable a warning: "Method 'a method' is abstract in class 'OptimizationAlgorithm' but
# is not overridden (abstract-method) since this class is not intended for instantiation
# pylint: disable=W0223
class MultiStartOptimizer(OptimizationAlgorithm, ABC):
    """
    An abstract class that implements multi start optimization and should be sub-classed by
    other optimizers.
    """

    def __init__(self, trials: int = 1, clip: float = 100.) -> None:
        """
        Constructs an instance of this optimizer.

        Args:
            trials: The number of trials for multi-start method. The first trial is solved with
                the initial guess of zero. If more than one trial is specified then
                initial guesses are uniformly drawn from ``[lowerbound, upperbound]``
                with potential clipping.
            clip: Clipping parameter for the initial guesses in the multi-start method.
                If a variable is unbounded then the lower bound and/or upper bound are replaced
                with the ``-clip`` or ``clip`` values correspondingly for the initial guesses.
        """
        super().__init__()
        self._trials = trials
        self._clip = clip

    def multi_start_solve(self, minimize: Callable[[np.array], np.array],
                          problem: QuadraticProgram) -> OptimizationResult:
        """Applies a multi start method given a local optimizer.

        Args:
            minimize: A callable object that minimizes the problem specified
            problem: A problem to solve

        Returns:
            The result of the multi start algorithm applied to the problem.
        """
        fval_sol = INFINITY
        x_sol = None    # type: Optional[np.array]

        # Implementation of multi-start optimizer
        for trial in range(self._trials):
            x_0 = np.zeros(problem.get_num_vars())
            if trial > 0:
                for i, var in enumerate(problem.variables):
                    lowerbound = var.lowerbound if var.lowerbound > -INFINITY else -self._clip
                    upperbound = var.upperbound if var.upperbound < INFINITY else self._clip
                    x_0[i] = uniform.rvs(lowerbound, (upperbound - lowerbound))
            # run optimization
            t_0 = time.time()
            x = minimize(x_0)
            logger.debug("minimize done in: %s seconds", str(time.time() - t_0))

            # we minimize, to get actual objective value we must multiply by the sense value
            fval = problem.objective.evaluate(x) * problem.objective.sense.value
            # we minimize the objective
            if fval < fval_sol:
                # here we get back to the original sense of the problem
                fval_sol = fval * problem.objective.sense.value
                x_sol = x

        return OptimizationResult(x_sol, fval_sol, x_sol, variables=problem.variables)

    @property
    def trials(self) -> int:
        """ Returns the number of trials for this optimizer.

        Returns:
            The number of trials.
        """
        return self._trials

    @trials.setter
    def trials(self, trials: int) -> None:
        """Sets the number of trials.

        Args:
            trials: The number of trials to set.
        """
        self._trials = trials

    @property
    def clip(self) -> float:
        """ Returns the clip value for this optimizer.

        Returns:
            The clip value.
        """
        return self._clip

    @clip.setter
    def clip(self, clip: float) -> None:
        """Sets the clip value.

        Args:
            clip: The clip value to set.
        """
        self._clip = clip
