
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

"""A recursive minimal eigen optimizer in Qiskit Optimization.

    Examples:
        >>> problem = OptimizationProblem()
        >>> # specify problem here
        >>> # specify minimum eigen solver to be used, e.g., QAOA
        >>> qaoa = QAOA(...)
        >>> optimizer = RecursiveMinEigenOptimizer(qaoa)
        >>> result = optimizer.solve(problem)
"""

from copy import deepcopy
from typing import Optional
import numpy as np
from cplex import SparseTriple

from qiskit.optimization import QiskitOptimizationError
from qiskit.optimization.algorithms import OptimizationAlgorithm, MinimumEigenOptimizer
from qiskit.optimization.problems import OptimizationProblem
from qiskit.optimization.results import OptimizationResult
from qiskit.optimization.converters import (PenalizeLinearEqualityConstraints,
                                            IntegerToBinaryConverter)
from qiskit.aqua.algorithms import ClassicalMinimumEigensolver


class RecursiveMinimumEigenOptimizer(OptimizationAlgorithm):
    """
    TODO
    """

    def __init__(self, min_eigen_optimizer: MinimumEigenOptimizer, min_num_vars: int = 1,
                 min_num_vars_optimizer: Optional[OptimizationAlgorithm] = None,
                 penalty: Optional[float] = None) -> None:
        """
        TODO: add flag to store full history...
        """
        # TODO: should also allow function that maps problem to <ZZ>-correlators?
        #  --> would support efficient classical implementation for QAOA with depth p=1
        self._min_eigen_optimizer = min_eigen_optimizer
        if min_num_vars < 1:
            raise QiskitOptimizationError('Minimal problem size needs to be >= 1!')
        self._min_num_vars = min_num_vars
        if min_num_vars_optimizer:
            self._min_num_vars_optimizer = min_num_vars_optimizer
        else:
            self._min_num_vars_optimizer = MinimumEigenOptimizer(ClassicalMinimumEigensolver())
        self._penalty = penalty

    def is_compatible(self, problem: OptimizationProblem) -> Optional[str]:
        """
        TODO
        """
        return None

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """
        # handle variable replacements via variable names
        # --> allows to easily adjust problems and roll-out final results
        # TODO
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
        problem_ref = deepcopy(problem_)

        # run recursive optimization until the resulting problem is small enough
        replacements = {}
        while problem_.variables.get_num() > self._min_num_vars:

            # solve current problem with optimizer
            result = self._min_eigen_optimizer.solve(problem_)
            details = result.results[0]

            # analyze results to get strongest correlation
            states = [v[0] for v in details]
            probs = [v[2] for v in details]
            correlations = self._construct_correlations(states, probs)
            i, j = self._find_strongest_correlation(correlations)

            xi = problem_.variables.get_names(i)
            xj = problem_.variables.get_names(j)
            if correlations[i, j] > 0:
                problem_ = problem_.substitute_variables(variables=SparseTriple([i], [j], [1]))
                replacements[xi] = (xj, 1)
            else:
                problem_ = problem_.substitute_variables(variables=SparseTriple([i], [j], [-1]))
                replacements[xi] = (xj, -1)

        # solve remaining problem
        result = self._min_num_vars_optimizer.solve(problem_)

        # unroll replacements
        var_values = {}
        for i, name in enumerate(problem_.variables.get_names()):
            var_values[name] = result.x[i]

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
        for xi in problem_ref.variables.get_names():
            if xi not in var_values:
                find_value(xi, replacements, var_values)

        # construct result
        x = [var_values[name] for name in problem_ref.variables.get_names()]
        fval = 0  # TODO: problem.objective.evaluate(x)
        results = OptimizationResult(x, fval, (replacements, int_to_bin_converter))
        results = int_to_bin_converter.decode(results)
        return results

    def _construct_correlations(self, states, probs):
        n = len(states[0])
        correlations = np.zeros((n, n))
        for k, p in enumerate(probs):
            b = states[k][::-1]
            for i in range(n):
                for j in range(i):
                    if b[i] == b[j]:
                        correlations[i, j] += p
                    else:
                        correlations[i, j] -= p
        return correlations

    def _find_strongest_correlation(self, M):
        m_max = np.argmax(np.abs(M.flatten()))
        i = m_max // len(M)
        j = m_max - i*len(M)
        return (i, j)
