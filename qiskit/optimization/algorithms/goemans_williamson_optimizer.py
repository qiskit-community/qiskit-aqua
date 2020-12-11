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

from typing import Optional, List, Tuple, Union

import numpy as np
from qiskit.aqua import MissingOptionalLibraryError
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import OptimizationAlgorithm, OptimizationResult, \
    OptimizationResultStatus
from qiskit.optimization.applications.ising.max_cut import cut_value
from qiskit.optimization.problems import Variable

try:
    import cvxpy as cvx
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


class GoemansWilliamsonOptimizationResult(OptimizationResult):
    """
    Contains results of the Goemans-Williamson algorithm. The properties ``x`` and ``fval`` contain
    values of just one solution. Explore ``all_solution`` for all possible solutions.
    """
    def __init__(self, x: Optional[Union[List[float], np.ndarray]], fval: float,
                 variables: List[Variable], status: OptimizationResultStatus,
                 all_solutions: Optional[List[Tuple[np.ndarray, float]]],
                 sdp_solution: np.ndarray) -> None:
        """
        Args:
            x: the optimal value found in the optimization.
            fval: the optimal function value.
            variables: the list of variables of the optimization problem.
            status: the termination status of the optimization algorithm.
            all_solutions: all solutions.
            sdp_solution: an SDP solution of the problem.

        Raises:
            MissingOptionalLibraryError: CVXPY is not installed.
        """
        if not _HAS_CVXPY:
            raise MissingOptionalLibraryError(
                libname='CVXPY',
                name='GoemansWilliamsonOptimizer',
                pip_install='pip install qiskit-aqua[cvxpy]')

        super().__init__(x, fval, variables, status, None)
        self._all_solutions = all_solutions
        self._sdp_solution = sdp_solution

    @property
    def explored_solutions(self) -> Optional[List[Tuple[np.ndarray, float]]]:
        """
        Returns:
            All generated solutions and their values.

        """
        return self._all_solutions

    @property
    def sdp_solution(self) -> np.ndarray:
        """
        Returns:
            Returns an SDP solution of the problem.
        """
        return self._sdp_solution


class GoemansWilliamsonOptimizer(OptimizationAlgorithm):
    """
    Goemans-Williamson algorithm to approximate the max-cut of a problem.
    The quadratic program for max-cut is given by:

    max sum_{i,j<i} w[i,j]*x[i]*(1-x[j])

    Therefore the quadratic term encodes the negative of the adjacency matrix of
    the graph.
    """

    def __init__(self, num_cuts: int, sort_cuts: bool = True,
                 unique_cuts: bool = True, seed: int = 0):
        """
        Args:
            num_cuts: Number of cuts to generate.
            sort_cuts: True if sort cuts by their values.
            unique_cuts: The solve method returns only unique cuts, thus there may be less cuts
                than ``num_cuts``.
            seed: A seed value for the random number generator.
        """
        super().__init__()

        self._num_cuts = num_cuts
        self._sort_cuts = sort_cuts
        self._unique_cuts = unique_cuts
        np.random.seed(seed)

    # todo: implement
    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with the optimizer implementing this method.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns the incompatibility message. If the message is empty no issues were found.
        """
        raise NotImplementedError

    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """
        Returns a list of cuts generated according to the Goemans-Williamson algorithm.

        Args:
            problem: The quadratic problem that encodes the max-cut problem.

        Returns:
            cuts: A list of generated cuts.
        """
        adj_matrix = self._extract_adjacency_matrix(problem)

        chi = self._solve_max_cut_sdp(adj_matrix)

        cuts = self._generate_random_cuts(chi, len(adj_matrix))

        solutions = [(cuts[i, :], cut_value(cuts[i, :], adj_matrix)) for i in range(self._num_cuts)]

        if self._sort_cuts:
            solutions.sort(key=lambda x: -x[1])

        if self._unique_cuts:
            solutions = self._get_unique_cuts(solutions)

        solutions = solutions[:self._num_cuts]
        return GoemansWilliamsonOptimizationResult(x=solutions[0][0],
                                                   fval=solutions[0][1],
                                                   variables=problem.variables,
                                                   status=OptimizationResultStatus.SUCCESS,
                                                   all_solutions=solutions,
                                                   sdp_solution=chi)

    def _get_unique_cuts(self, solutions: List[Tuple[np.ndarray, float]]) \
            -> List[Tuple[np.ndarray, float]]:
        """
        Returns:
            Unique Goemans-Williamson cuts.
        """

        # Remove symmetry in the cuts to chose the unique ones.
        # Cuts 010 and 101 are symmetric(same cut), so we convert all cuts
        # starting from 1 to start from 0. In the next loop repetitive cuts will be removed.
        for idx, cut in enumerate(solutions):
            if cut[0][0] == 1:
                solutions[idx] = (np.array([0 if _ == 1 else 1 for _ in cut[0]]), cut[1])

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
        adj_matrix = -problem.objective.quadratic.coefficients.toarray()
        adj_matrix = (adj_matrix + adj_matrix.T) / 2

        return adj_matrix

    def _solve_max_cut_sdp(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the maximum weight cut by generating |V| vectors with a vector program,
        then generating a random plane that cuts the vertices. This is the Goemans-Williamson
        algorithm that gives a .878-approximation.

        Returns:
            chi: a list of length |V| where the i-th element is +1 or -1, representing which
                set the it-h vertex is in. Returns None if an error occurs.

        Raises:
            MissingOptionalLibraryError: if CVXPY is not installed.
        """
        num_vertices = len(adj_matrix)
        constraints, expr = [], 0

        # variables
        x = cvx.Variable((num_vertices, num_vertices), PSD=True)

        # constraints
        for i in range(num_vertices):
            constraints.append(x[i, i] == 1)

        # objective function
        expr = cvx.sum(cvx.multiply(adj_matrix, (np.ones((num_vertices, num_vertices)) - x)))

        # solve
        problem = cvx.Problem(cvx.Maximize(expr), constraints)
        problem.solve()

        # todo: add checks that the problem is solved
        return x.value

    def _generate_random_cuts(self, chi: np.ndarray, num_vertices: int) -> np.ndarray:
        """
        Random hyperplane partitions vertices.

        Args:
            chi: a list of length |V| where the i-th element is +1 or -1, representing
                which set the i-th vertex is in.
            num_vertices: the number of vertices in the graph

        Returns:
            An array of random cuts.
        """
        eigenvalues = np.linalg.eigh(chi)[0]
        # todo: weird numbers: 1.001 and 0.00001
        if min(eigenvalues) < 0:
            chi = chi + (1.001 * abs(min(eigenvalues)) * np.identity(num_vertices))
        elif min(eigenvalues) == 0:
            chi = chi + 0.00001 * np.identity(num_vertices)
        x = np.linalg.cholesky(chi).T

        r = np.random.normal(size=(self._num_cuts, num_vertices))

        # todo: why "+ 0" ?
        return (np.dot(r, x) > 0) + 0
