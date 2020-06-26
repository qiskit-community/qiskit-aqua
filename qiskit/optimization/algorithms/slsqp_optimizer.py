
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
from typing import Optional, List, cast

import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.stats import truncnorm
import time

from .optimization_algorithm import OptimizationAlgorithm, OptimizationResult
from ..exceptions import QiskitOptimizationError

from ..problems.quadratic_program import QuadraticProgram
from ..problems.constraint import Constraint


logger = logging.getLogger(__name__)


class SlsqpOptimizer(OptimizationAlgorithm):
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
        >>> optimizer = SlsqpOptimizer()
        >>> result = optimizer.solve(problem)
    """

    def __init__(self, iter=100, acc=1.0E-6, iprint=0, shots=1) -> None:
        """Initializes the SlsqpOptimizer.

        This initializer takes the algorithmic parameters of COBYLA and stores them for later use
        of ``fmin_cobyla`` when :meth:`solve` is invoked.
        This optimizer can be applied to find a (local) optimum for problems consisting of only
        continuous variables.

        Args:
            iter:  number of iteration
            acc: accuracy
            iprint: print option (0,1,2)
            shots: number of shots for multi-start method
        """

        self._iter = iter
        self._acc = acc
        self._iprint = iprint
        self._shots = shots
        self._log = logging.getLogger(__name__)

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
        # check compatibility and raise exception if incompatible
        msg = self.get_compatibility_msg(problem)
        if len(msg) > 0:
            raise QiskitOptimizationError('Incompatible problem: {}'.format(msg))

        # construct quadratic objective function
        def objective(x):
            return problem.objective.sense.value * problem.objective.evaluate(x)

        def objective_gradient(x):
            return problem.objective.sense.value * problem.objective.evaluate_gradient(x)

        # initialize constraints list
        slsqp_bounds = []
        slsqp_eq_constraints = []
        slsqp_ineq_constraints = []

        # add lower/upper bound constraints
        for variable in problem.variables:
            lowerbound = variable.lowerbound
            upperbound = variable.upperbound
            slsqp_bounds.append((lowerbound, upperbound))

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

        shots = self._shots
        fval = 1e20
        # todo: we don't need array, we need a variable and var type definition here
        x = np.zeros(problem.get_num_vars())

        # Implementation of multi-start SLSQP optimizer
        for shot in range(shots):
            x_0 = np.zeros(len(problem.variables))
            if shot > 0:  # give the zero vector also a chance
                for i, variable in enumerate(problem.variables):
                    lowerbound = variable.lowerbound
                    upperbound = variable.upperbound
                    x_0[i] = truncnorm.rvs(lowerbound, upperbound)

            # run optimization
            t0 = time.time()
            xs = fmin_slsqp(objective, x_0, eqcons=slsqp_eq_constraints,
                            ieqcons=slsqp_ineq_constraints,
                            bounds=slsqp_bounds, fprime=objective_gradient, iter=self._iter,
                            acc=self._acc, iprint=self._iprint)
            self._log.debug("SLSQP done in: %s seconds", str(time.time() - t0))
            fvals = problem.objective.sense.value * objective(xs)
            if fvals < fval:
                fval = fvals
                x = xs

        print('Final value: ' + str(fval))

        # return results
        return OptimizationResult(x, fval, x)
