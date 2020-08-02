
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

"""A wrapper for minimum eigen solvers from Aqua to be used within the optimization module."""

from typing import Optional, Any, Union, Tuple, List, cast
import numpy as np

from qiskit.aqua.algorithms import MinimumEigensolver
from qiskit.aqua.operators import StateFn, DictStateFn

from .optimization_algorithm import OptimizationAlgorithm, OptimizationResult
from ..problems.quadratic_program import QuadraticProgram
from ..problems.variable import Variable
from ..converters.quadratic_program_to_ising import QuadraticProgramToIsing
from ..converters.quadratic_program_to_qubo import QuadraticProgramToQubo


class MinimumEigenOptimizerResult(OptimizationResult):
    """ Minimum Eigen Optimizer Result."""

    def __init__(self, x: Optional[Any] = None, fval: Optional[Any] = None,
                 samples: Optional[Any] = None, results: Optional[Any] = None,
                 variables: Optional[List[Variable]] = None) -> None:
        super().__init__(x, fval, results, variables=variables)
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
    """A wrapper for minimum eigen solvers from Qiskit Aqua.

    This class provides a wrapper for minimum eigen solvers from Qiskit to be used within
    the optimization module.
    It assumes a problem consisting only of binary or integer variables as well as linear equality
    constraints thereof. It converts such a problem into a Quadratic Unconstrained Binary
    Optimization (QUBO) problem by expanding integer variables into binary variables and by adding
    the linear equality constraints as weighted penalty terms to the objective function. The
    resulting QUBO is then translated into an Ising Hamiltonian whose minimal eigen vector and
    corresponding eigenstate correspond to the optimal solution of the original optimization
    problem. The provided minimum eigen solver is then used to approximate the ground state of the
    Hamiltonian to find a good solution for the optimization problem.

    Examples:
        Outline of how to use this class:

    .. code-block::

        from qiskit.aqua.algorithms import QAOA
        from qiskit.optimization.problems import QuadraticProgram
        from qiskit.optimization.algorithms import MinimumEigenOptimizer
        problem = QuadraticProgram()
        # specify problem here
        # specify minimum eigen solver to be used, e.g., QAOA
        qaoa = QAOA(...)
        optimizer = MinimumEigenOptimizer(qaoa)
        result = optimizer.solve(problem)
    """

    def __init__(self, min_eigen_solver: MinimumEigensolver, penalty: Optional[float] = None
                 ) -> None:
        """
        This initializer takes the minimum eigen solver to be used to approximate the ground state
        of the resulting Hamiltonian as well as a optional penalty factor to scale penalty terms
        representing linear equality constraints. If no penalty factor is provided, a default
        is computed during the algorithm (TODO).

        Args:
            min_eigen_solver: The eigen solver to find the ground state of the Hamiltonian.
            penalty: The penalty factor to be used, or ``None`` for applying a default logic.
        """
        self._min_eigen_solver = min_eigen_solver
        self._penalty = penalty

    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem can be converted
        to a QUBO, and otherwise, returns a message explaining the incompatibility.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            A message describing the incompatibility.
        """
        return QuadraticProgramToQubo.get_compatibility_msg(problem)

    def solve(self, problem: QuadraticProgram) -> MinimumEigenOptimizerResult:
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If problem not compatible.
        """
        self._verify_compatibility(problem)

        # convert problem to QUBO
        qubo_converter = QuadraticProgramToQubo()
        problem_ = qubo_converter.encode(problem)

        # construct operator and offset
        operator_converter = QuadraticProgramToIsing()
        operator, offset = operator_converter.encode(problem_)

        # only try to solve non-empty Ising Hamiltonians
        x = None  # type: Optional[Any]
        if operator.num_qubits > 0:

            # approximate ground state of operator using min eigen solver
            eigen_results = self._min_eigen_solver.compute_minimum_eigenvalue(operator)

            # analyze results
            # backend = getattr(self._min_eigen_solver, 'quantum_instance', None)
            samples = _eigenvector_to_solutions(eigen_results.eigenstate, problem_)
            # print(offset, samples)
            # samples = [(res[0], problem_.objective.sense.value * (res[1] + offset), res[2])
            #    for res in samples]
            samples.sort(key=lambda x: problem_.objective.sense.value * x[1])
            x = samples[0][0]
            fval = samples[0][1]

        # if Hamiltonian is empty, then the objective function is constant to the offset
        else:
            x = [0]*problem_.get_num_binary_vars()
            fval = offset
            x_str = '0'*problem_.get_num_binary_vars()
            samples = [(x_str, offset, 1.0)]

        # translate result back to integers
        opt_res = MinimumEigenOptimizerResult(x, fval, samples, qubo_converter,
                                              variables=problem.variables)
        opt_res = cast(MinimumEigenOptimizerResult, qubo_converter.decode(opt_res))

        # translate results back to original problem
        return opt_res


def _eigenvector_to_solutions(eigenvector: Union[dict, np.ndarray, StateFn],
                              qubo: QuadraticProgram,
                              min_probability: float = 1e-6,
                              ) -> List[Tuple[str, float, float]]:
    """Convert the eigenvector to the bitstrings and corresponding eigenvalues.

    Args:
        eigenvector: The eigenvector from which the solution states are extracted.
        qubo: The QUBO to evaluate at the bitstring.
        min_probability: Only consider states where the amplitude exceeds this threshold.

    Returns:
        For each computational basis state contained in the eigenvector, return the basis
        state as bitstring along with the QUBO evaluated at that bitstring and the
        probability of sampling this bitstring from the eigenvector.

    Examples:
        >>> op = MatrixOp(numpy.array([[1, 1], [1, -1]]) / numpy.sqrt(2))
        >>> eigenvectors = {'0': 12, '1': 1}
        >>> print(eigenvector_to_solutions(eigenvectors, op))
        [('0', 0.7071067811865475, 0.9230769230769231),
        ('1', -0.7071067811865475, 0.07692307692307693)]

        >>> op = MatrixOp(numpy.array([[1, 1], [1, -1]]) / numpy.sqrt(2))
        >>> eigenvectors = numpy.array([1, 1] / numpy.sqrt(2), dtype=complex)
        >>> print(eigenvector_to_solutions(eigenvectors, op))
        [('0', 0.7071067811865475, 0.4999999999999999),
        ('1', -0.7071067811865475, 0.4999999999999999)]

    Raises:
        TypeError: If the type of eigenvector is not supported.
    """
    if isinstance(eigenvector, DictStateFn):
        eigenvector = {bitstr: val**2 for (bitstr, val) in eigenvector.primitive.items()}
    elif isinstance(eigenvector, StateFn):
        eigenvector = eigenvector.to_matrix()

    solutions = []
    if isinstance(eigenvector, dict):
        all_counts = sum(eigenvector.values())
        # iterate over all samples
        for bitstr, count in eigenvector.items():
            sampling_probability = count / all_counts
            # add the bitstring, if the sampling probability exceeds the threshold
            if sampling_probability > 0:
                if sampling_probability >= min_probability:
                    value = qubo.objective.evaluate([int(bit) for bit in bitstr])
                    solutions += [(bitstr, value, sampling_probability)]

    elif isinstance(eigenvector, np.ndarray):
        num_qubits = int(np.log2(eigenvector.size))
        probabilities = np.abs(eigenvector * eigenvector.conj())

        # iterate over all states and their sampling probabilities
        for i, sampling_probability in enumerate(probabilities):

            # add the i-th state if the sampling probability exceeds the threshold
            if sampling_probability > 0:
                if sampling_probability >= min_probability:
                    bitstr = '{:b}'.format(i).rjust(num_qubits, '0')[::-1]
                    value = qubo.objective.evaluate([int(bit) for bit in bitstr])
                    solutions += [(bitstr, value, sampling_probability)]

    else:
        raise TypeError('Unsupported format of eigenvector. Provide a dict or numpy.ndarray.')

    return solutions
