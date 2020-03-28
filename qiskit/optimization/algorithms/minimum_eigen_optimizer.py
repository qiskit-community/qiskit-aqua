
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

"""A wrapper for minimum eigen solvers from Qiskit Aqua to be used within Qiskit Optimization.

Examples:
    >>> problem = OptimizationProblem()
    >>> # specify problem here
    >>> # specify minimum eigen solver to be used, e.g., QAOA
    >>> qaoa = QAOA(...)
    >>> optimizer = MinEigenOptimizer(qaoa)
    >>> result = optimizer.solve(problem)
"""

from typing import Optional, Any
import numpy as np

from qiskit.aqua.algorithms import MinimumEigensolver

from .optimization_algorithm import OptimizationAlgorithm
from ..problems.optimization_problem import OptimizationProblem
from ..utils.eigenvector_to_solutions import eigenvector_to_solutions
from ..converters.optimization_problem_to_operator import OptimizationProblemToOperator
from ..converters.optimization_problem_to_qubo import OptimizationProblemToQubo
from ..results.optimization_result import OptimizationResult


class MinimumEigenOptimizerResult(OptimizationResult):
    """ Minimum Eigen Optimizer Result."""

    def __init__(self, x: Optional[Any] = None, fval: Optional[Any] = None,
                 samples: Optional[Any] = None, results: Optional[Any] = None) -> None:
        super().__init__(x, fval, results)
        self._samples = samples

    @property
    def samples(self) -> Any:
        """ returns samples """
        return self._samples

    @samples.setter
    def samples(self, samples: Any) -> None:
        """ set samples """
        self._samples = samples

    def get_correlations(self):
        """ get <Zi x Zj> correlation matrix from samples """

        states = [v[0] for v in self.samples]
        probs = [v[2] for v in self.samples]

        n = len(states[0])
        correlations = np.zeros((n, n))
        for k, prob in enumerate(probs):
            b = states[k]
            for i in range(n):
                for j in range(i):
                    if b[i] == b[j]:
                        correlations[i, j] += prob
                    else:
                        correlations[i, j] -= prob
        return correlations


class MinimumEigenOptimizer(OptimizationAlgorithm):
    """A wrapper for minimum eigen solvers from Qiskit Aqua to be used within Qiskit Optimization.

    This class provides a wrapper for minimum eigen solvers from Qiskit Aqua.
    It assumes a problem consisting only of binary or integer variables as well as linear equality
    constraints thereof. It converts such a problem into a Quadratic Unconstrained Binary
    Optimization (QUBO) problem by expanding integer variables into binary variables and by adding
    the linear equality constraints as weighted penalty terms to the objective function. The
    resulting QUBO is then translated into an Ising Hamiltonian whose minimal eigen vector and
    corresponding eigenstate correspond to the optimal solution of the original optimization
    problem. The provided minimum eigen solver is then used to approximate the groundstate of the
    Hamiltonian to find a good solution for the optimization problem.
    """

    def __init__(self, min_eigen_solver: MinimumEigensolver, penalty: Optional[float] = None
                 ) -> None:
        """Initializes the minimum eigen optimizer.

        This initializer takes the minimum eigen solver to be used to approximate the groundstate
        of the resulting Hamiltonian as well as a optional penalty factor to scale penalty terms
        representing linear equality constraints. If no penalty factor is provided, a default
        is computed during the algorithm (TODO).

        Args:
            min_eigen_solver: The eigen solver to find the groundstate of the Hamiltonian.
            penalty: The penalty factor to be used, or ``None`` for applying a default logic.
        """
        self._min_eigen_solver = min_eigen_solver
        self._penalty = penalty

    def is_compatible(self, problem: OptimizationProblem) -> Optional[str]:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem can be converted
        to a QUBO, and otherwise, returns a message explaining the incompatibility.

        Args:
            problem: The optization problem to check compatibility.

        Returns:
            Returns ``None`` if the problem is compatible and else a string with the error message.
        """
        return OptimizationProblemToQubo.is_compatible(problem)

    def solve(self, problem: OptimizationProblem) -> MinimumEigenOptimizerResult:
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        """

        # convert problem to QUBO
        qubo_converter = OptimizationProblemToQubo()
        problem_ = qubo_converter.encode(problem)

        # construct operator and offset
        operator_converter = OptimizationProblemToOperator()
        operator, offset = operator_converter.encode(problem_)

        # approximate ground state of operator using min eigen solver
        eigen_results = self._min_eigen_solver.compute_minimum_eigenvalue(operator)

        # analyze results
        samples = eigenvector_to_solutions(eigen_results.eigenstate, operator)
        samples = [(res[0], problem_.objective.get_sense() * (res[1] + offset), res[2])
                   for res in samples]
        samples.sort(key=lambda x: problem_.objective.get_sense() * x[1])

        # translate result back to integers
        opt_res = MinimumEigenOptimizerResult(samples[0][0], samples[0][1], samples, qubo_converter)
        opt_res = qubo_converter.decode(opt_res)

        # translate results back to original problem
        return opt_res
