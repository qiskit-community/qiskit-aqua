# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The COBYLA optimizer wrapped to be used within Qiskit's optimization module."""

from typing import Optional, cast, List, Tuple, Any

import numpy as np
from scipy.optimize import fmin_cobyla

from .multistart_optimizer import MultiStartOptimizer
from .optimization_algorithm import OptimizationResult
from ..exceptions import QiskitOptimizationError
from ..infinity import INFINITY
from ..problems.constraint import Constraint
from ..problems.quadratic_program import QuadraticProgram


class CobylaOptimizer(MultiStartOptimizer):
    """The SciPy COBYLA optimizer wrapped as an Qiskit :class:`OptimizationAlgorithm`.

    This class provides a wrapper for ``scipy.optimize.fmin_cobyla``
    (https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_cobyla.html)
    to be used within the optimization module.
    The arguments for ``fmin_cobyla`` are passed via the constructor.

    Examples:
        >>> from qiskit.optimization.problems import QuadraticProgram
        >>> from qiskit.optimization.algorithms import CobylaOptimizer
        >>> problem = QuadraticProgram()
        >>> # specify problem here
        >>> optimizer = CobylaOptimizer()
        >>> result = optimizer.solve(problem)
    """

    def __init__(self, rhobeg: float = 1.0, rhoend: float = 1e-4, maxfun: int = 1000,
                 disp: Optional[int] = None, catol: float = 2e-4, trials: int = 1,
                 clip: float = 100.) -> None:
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
            trials: The number of trials for multi-start method. The first trial is solved with
                the initial guess of zero. If more than one trial is specified then
                initial guesses are uniformly drawn from ``[lowerbound, upperbound]``
                with potential clipping.
            clip: Clipping parameter for the initial guesses in the multi-start method.
                If a variable is unbounded then the lower bound and/or upper bound are replaced
                with the ``-clip`` or ``clip`` values correspondingly for the initial guesses.
        """

        super().__init__(trials, clip)
        self._rhobeg = rhobeg
        self._rhoend = rhoend
        self._maxfun = maxfun
        self._disp = disp
        self._catol = catol

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
            return 'The COBYLA optimizer supports only continuous variables'

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
        def objective(x):
            return problem.objective.sense.value * problem.objective.evaluate(x)

        # initialize constraints list
        constraints = []

        # add lower/upper bound constraints
        for i, variable in enumerate(problem.variables):
            lowerbound = variable.lowerbound
            upperbound = variable.upperbound
            if lowerbound > -INFINITY:
                def lb_constraint(x, l_b=lowerbound, j=i):
                    return x[j] - l_b
                constraints += [lb_constraint]
            if upperbound < INFINITY:
                def ub_constraint(x, u_b=upperbound, j=i):
                    return u_b - x[j]
                constraints += [ub_constraint]

        # pylint: disable=no-member
        # add linear and quadratic constraints
        for constraint in cast(List[Constraint], problem.linear_constraints) +\
                cast(List[Constraint], problem.quadratic_constraints):
            rhs = constraint.rhs
            sense = constraint.sense

            if sense == Constraint.Sense.EQ:
                constraints += [
                    lambda x, rhs=rhs, c=constraint: rhs - c.evaluate(x),
                    lambda x, rhs=rhs, c=constraint: c.evaluate(x) - rhs
                ]
            elif sense == Constraint.Sense.LE:
                constraints += [lambda x, rhs=rhs, c=constraint: rhs - c.evaluate(x)]
            elif sense == Constraint.Sense.GE:
                constraints += [lambda x, rhs=rhs, c=constraint: c.evaluate(x) - rhs]
            else:
                raise QiskitOptimizationError('Unsupported constraint type!')

        # actual minimization function to be called by multi_start_solve
        def _minimize(x_0: np.ndarray) -> Tuple[np.ndarray, Any]:
            x = fmin_cobyla(objective, x_0, constraints, rhobeg=self._rhobeg,
                            rhoend=self._rhoend, maxfun=self._maxfun, disp=self._disp,
                            catol=self._catol)
            return x, None

        return self.multi_start_solve(_minimize, problem)
