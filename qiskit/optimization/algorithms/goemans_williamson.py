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

"""
Implementation of the Goemans-Williamson algorithm as an optimizer.
Requires CVXPY to run.
"""

from typing import Optional, List, Tuple

import numpy as np

from qiskit.aqua import MissingOptionalLibraryError
from qiskit.optimization.algorithms import OptimizationAlgorithm, OptimizationResult

from qiskit.optimization import QuadraticProgram
from qiskit.optimization.applications.ising.max_cut import cut_value


class GoemansWilliamsonOptimizer(OptimizationAlgorithm):
    """
    Goemans-Williamson algorithm to approximate the max-cut of a problem.
    The quadratic program for max-cut is given by:

    max sum_{i,j<i} w[i,j]*x[i]*(1-x[j])

    Therefore the quadratic term encodes the negative of the adjacency matrix of
    the graph.
    """

    def __init__(self, num_cuts: int, num_best: int = None, sort_cuts: bool = True,
                 unique_cuts: bool = True):
        """
        Args:
            num_cuts: Number of cuts to generate.
            num_best: number of best cuts to return. If None, all are returned.
            unique_cuts: The solve method returns only unique cuts.
        """
        super().__init__()
        self._num_cuts = num_cuts
        self._graph = None  # type: Optional[np.array]
        self._num_vertices = None
        self._sort_cuts = sort_cuts
        self._unique_cuts = unique_cuts
        self._num_best = num_best or num_cuts

    # todo: do we need property for the graph?
    @property
    def graph(self):
        """
        Graph of the problem as an adjacency matrix. No multi-edges or directed
        edges are allowed.
        """
        return self._graph

    @graph.setter
    def graph(self, graph: np.array):
        # todo: check that array is square
        self._graph = graph
        self._num_vertices = len(graph)

    # todo: implement
    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with the optimizer implementing this method.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns the incompatibility message. If the message is empty no issues were found.
        """
        raise NotImplemented

    # todo: define Tuple types
    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """
        Returns a list of cuts generated according to the Goemans-Williamson algorithm.

        Args:
            problem: The quadratic problem that encodes the max-cut problem.

        Returns:
            cuts: A list of generated cuts.
        """
        self._graph = self._extract_adjacency_matrix(problem)

        chi = self._solve_max_cut_sdp()

        cuts = self._generate_random_cuts(chi)

        solutions = [(cuts[i, :], cut_value(cuts[i, :], self.graph)) for i in range(self._num_cuts)]

        if self._sort_cuts:
            solutions.sort(key=lambda x: -x[1])

        if self._unique_cuts:
            solutions = self._get_unique_cuts(solutions)

        # this is List[Tuple]
        return solutions[:self._num_best]

    def _get_unique_cuts(self, solutions: List[Tuple]) -> List[Tuple]:
        """
        Returns:
            Unique GW cuts.
        """

        # Remove symmetry in the cuts to chose the unique ones.
        for idx, cut in enumerate(solutions):
            if cut[0][0] == 1:
                solutions[idx] = ([0 if _ == 1 else 1 for _ in cut[0]], cut[1])

        seen_cuts = set()
        unique_cuts = []
        for cut in solutions:
            cut_str = ''.join([str(_) for _ in cut[0]])
            if cut_str in seen_cuts:
                continue

            seen_cuts.add(cut_str)
            unique_cuts.append(cut)

        return unique_cuts

    @staticmethod
    def _extract_adjacency_matrix(problem: QuadraticProgram) -> np.array:
        """
        Extracts the adjacency matrix from the given quadratic program.

        Args:
            problem: A QuadraticProgram describing the max-cut optimization problem.

        Returns:
            adjacency matrix of the graph.
        """
        graph = -problem.objective.quadratic.coefficients.toarray()
        graph = (graph + graph.T) / 2

        return graph

    def _solve_max_cut_sdp(self) -> np.array:
        """
        Calculates the maximum weight cut by generating |V| vectors with a vector program,
        then generating a random plane that cuts the vertices. This is the Goemans-Williamson
        algorithm that gives a .878-approximation.

        Returns:
            chi: a list of length |V| where the ith element is +1 or -1, representing which
                set the ith vertex is in. Returns None if an error occurs.
        """
        try:
            import cvxpy as cvx
        except ImportError:
            raise MissingOptionalLibraryError(
                libname='CVXPY',
                name='GoemansWilliamsonOptimizer',
                pip_install='pip install qiskit-aqua[cvxpy]')

        constraints, expr = [], 0

        # variables
        x = cvx.Variable((self._num_vertices, self._num_vertices), PSD=True)

        # constraints
        for i in range(self._num_vertices):
            constraints.append(x[i, i] == 1)

        # objective function
        expr = cvx.sum(cvx.multiply(self._graph, (np.ones((self._num_vertices, self._num_vertices)) - x)))

        # solve
        problem = cvx.Problem(cvx.Maximize(expr), constraints)
        # print(problem)
        problem.solve()

        # todo: add checks that the problem is solved
        return x.value

    def _generate_random_cuts(self, chi: np.array) -> np.array:
        """
        Random hyperplane partitions vertices.

        Args:
            chi: a list of length |V| where the ith element is +1 or -1, representing
            which set the ith vertex is in.
        """
        eigs = np.linalg.eigh(chi)[0]
        # todo: weird numbers: 1.001 and 0.00001
        if min(eigs) < 0:
            chi = chi + (1.001 * abs(min(eigs)) * np.identity(self._num_vertices))
        elif min(eigs) == 0:
            chi = chi + 0.00001 * np.identity(self._num_vertices)
        x = np.linalg.cholesky(chi).T

        r = np.random.normal(size=(self._num_cuts, self._num_vertices))

        # todo: why "+ 0" ?
        return (np.dot(r, x) > 0) + 0
