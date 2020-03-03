
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

from qiskit.optimization.algorithms import OptimizationAlgorithm


class RecursiveIsingOptimizer(OptimizationAlgorithm):
    """
    TODO
    """

    def __init__(self, ising_solver, mode='correlation', min_num_vars=0):
        """
        TODO
        """
        # TODO: should also allow function that maps problem to <ZZ>-correlators?
        #  --> would support efficient classical implementation for QAOA with depth p=1
        self._eigen_solver = eigen_solver  # TODO: base on eigen_solver or ising_optimizer?
        self._mode = mode
        self._min_num_vars = min_num_vars

    def solve(self, problem):
        # handle variable replacements via variable names
        # --> allows to easily adjust problems and roll-out final results
        # TODO
        pass
