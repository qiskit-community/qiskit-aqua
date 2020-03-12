
# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
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

from typing import Optional

from qiskit.aqua.algorithms import MinimumEigensolver
from qiskit.optimization.problems import OptimizationProblem
from qiskit.optimization.algorithms import OptimizationAlgorithm
from qiskit.optimization.utils import QiskitOptimizationError
from qiskit.optimization.converters import (OptimizationProblemToOperator,
                                            PenalizeLinearEqualityConstraints,
                                            IntegerToBinaryConverter)
from qiskit.optimization.utils import eigenvector_to_solutions
from qiskit.optimization.results import OptimizationResult


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

        Checks whether the given problem is compatible, i.e., whether the problem contains only
        binary and integer variables as well as linear equality constraints, and otherwise,
        returns a message explaining the incompatibility.

        Args:
            problem: The optization problem to check compatibility.

        Returns:
            Returns ``None`` if the problem is compatible and else a string with the error message.
        """

        # initialize message
        msg = ''

        # check whether there are incompatible variable types
        if problem.variables.get_num_continuous() > 0:
            msg += 'Continuous variables are not supported! '
        if problem.variables.get_num_semicontinuous() > 0:
            msg += 'Semi-continuous variables are not supported! '
        # if problem.variables.get_num_integer() > 0:
        #     # TODO: to be removed once integer to binary mapping is introduced
        #     msg += 'Integer variables are not supported! '
        if problem.variables.get_num_semiinteger() > 0:
            # TODO: to be removed once semi-integer to binary mapping is introduced
            msg += 'Semi-integer variables are not supported! '

        # check whether there are incompatible constraint types
        if not all([sense == 'E' for sense in problem.linear_constraints.get_senses()]):
            msg += 'Only linear equality constraints are supported.'

        # TODO: check for quadratic constraints

        # if an error occurred, return error message, otherwise, return None
        if len(msg) > 0:
            return msg.strip()
        else:
            return None

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

        # analyze compatibility of problem
        msg = self.is_compatible(problem)
        if msg is not None:
            raise QiskitOptimizationError('Incompatible problem: %s' % msg)

        # map integer variables to binary variables
        int_to_bin_converter = IntegerToBinaryConverter()
        problem_ = int_to_bin_converter.encode(problem)

        # penalize linear equality constraints with only binary variables
        if self._penalty is None:
            # TODO: should be derived from problem
            penalty = 1e5
        else:
            penalty = self._penalty
        lin_eq_converter = PenalizeLinearEqualityConstraints()
        problem_ = lin_eq_converter.encode(problem_, penalty_factor=penalty)

        # construct operator and offset
        operator_converter = OptimizationProblemToOperator()
        operator, offset = operator_converter.encode(problem_)

        # approximate ground state of operator using min eigen solver
        eigen_results = self._min_eigen_solver.compute_minimum_eigenvalue(operator)

        # analyze results
        results = eigenvector_to_solutions(eigen_results.eigenstate, operator)
        results = [(res[0], problem_.objective.get_sense() * (res[1] + offset), res[2])
                   for res in results]
        results.sort(key=lambda x: problem_.objective.get_sense() * x[1])

        # translate result back to integers
        opt_res = OptimizationResult(results[0][0], results[0][1], (results, int_to_bin_converter))
        opt_res = int_to_bin_converter.decode(opt_res)

        # translate results back to original problem
        return opt_res
