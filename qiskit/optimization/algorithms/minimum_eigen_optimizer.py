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

"""A wrapper for minimum eigen solvers from Aqua to be used within the optimization module."""
from typing import Optional, Any, Union, List, cast

import numpy as np

from qiskit.aqua.algorithms import MinimumEigensolver, MinimumEigensolverResult
from qiskit.aqua.operators import StateFn, DictStateFn
from .optimization_algorithm import (OptimizationResultStatus, OptimizationAlgorithm,
                                     OptimizationResult, SolutionSample)
from .. import QiskitOptimizationError
from ..converters.quadratic_program_to_qubo import QuadraticProgramToQubo, QuadraticProgramConverter
from ..problems.quadratic_program import QuadraticProgram, Variable


class MinimumEigenOptimizationResult(OptimizationResult):
    """ Minimum Eigen Optimizer Result."""

    def __init__(self, x: Union[List[float], np.ndarray], fval: float, variables: List[Variable],
                 status: OptimizationResultStatus,
                 samples: Optional[List[SolutionSample]] = None,
                 min_eigen_solver_result: Optional[MinimumEigensolverResult] = None,
                 raw_samples: Optional[List[SolutionSample]] = None) -> None:
        """
        Args:
            x: the optimal value found by ``MinimumEigensolver``.
            fval: the optimal function value.
            variables: the list of variables of the optimization problem.
            status: the termination status of the optimization algorithm.
            min_eigen_solver_result: the result obtained from the underlying algorithm.
            samples: the x value, the objective function value of the original problem,
                the probability, and the status of sampling.
            raw_samples: the x values of the QUBO, the objective function value of the QUBO,
                and the probability of sampling.
        """
        super().__init__(x=x, fval=fval, variables=variables, status=status, raw_results=None,
                         samples=samples)
        self._min_eigen_solver_result = min_eigen_solver_result
        self._raw_samples = raw_samples

    @property
    def min_eigen_solver_result(self) -> MinimumEigensolverResult:
        """Returns a result object obtained from the instance of :class:`MinimumEigensolver`."""
        return self._min_eigen_solver_result

    def get_correlations(self) -> np.ndarray:
        """Get <Zi x Zj> correlation matrix from samples."""

        states = [v.x for v in self.samples]
        probs = [v.probability for v in self.samples]

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

    @property
    def raw_samples(self) -> Optional[List[SolutionSample]]:
        """Returns the list of raw solution samples of ``MinimumEigensolver``.

        Returns:
            The list of raw solution samples of ``MinimumEigensolver``.
        """
        return self._raw_samples


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

    def __init__(self, min_eigen_solver: MinimumEigensolver, penalty: Optional[float] = None,
                 converters: Optional[Union[QuadraticProgramConverter,
                                            List[QuadraticProgramConverter]]] = None) -> None:
        """
        This initializer takes the minimum eigen solver to be used to approximate the ground state
        of the resulting Hamiltonian as well as a optional penalty factor to scale penalty terms
        representing linear equality constraints. If no penalty factor is provided, a default
        is computed during the algorithm (TODO).

        Args:
            min_eigen_solver: The eigen solver to find the ground state of the Hamiltonian.
            penalty: The penalty factor to be used, or ``None`` for applying a default logic.
            converters: The converters to use for converting a problem into a different form.
                By default, when None is specified, an internally created instance of
                :class:`~qiskit.optimization.converters.QuadraticProgramToQubo` will be used.

        Raises:
            TypeError: When one of converters has an invalid type.
            QiskitOptimizationError: When the minimum eigensolver does not return an eigenstate.
        """

        if not min_eigen_solver.supports_aux_operators():
            raise QiskitOptimizationError('Given MinimumEigensolver does not return the eigenstate '
                                          + 'and is not supported by the MinimumEigenOptimizer.')
        self._min_eigen_solver = min_eigen_solver
        self._penalty = penalty

        self._converters = self._prepare_converters(converters, penalty)

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

    @property
    def min_eigen_solver(self) -> MinimumEigensolver:
        """Returns the minimum eigensolver."""
        return self._min_eigen_solver

    @min_eigen_solver.setter
    def min_eigen_solver(self, min_eigen_solver: MinimumEigensolver) -> None:
        """Sets the minimum eigensolver."""
        self._min_eigen_solver = min_eigen_solver

    def solve(self, problem: QuadraticProgram) -> MinimumEigenOptimizationResult:
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
        problem_ = self._convert(problem, self._converters)

        # construct operator and offset
        operator, offset = problem_.to_ising()

        # only try to solve non-empty Ising Hamiltonians
        x = None  # type: Optional[Any]
        eigen_result = None  # type: MinimumEigensolverResult
        if operator.num_qubits > 0:

            # approximate ground state of operator using min eigen solver
            eigen_result = self._min_eigen_solver.compute_minimum_eigenvalue(operator)

            # analyze results
            # backend = getattr(self._min_eigen_solver, 'quantum_instance', None)
            fval = None
            x = None
            raw_samples = None
            if eigen_result.eigenstate is not None:
                raw_samples = _eigenvector_to_solutions(eigen_result.eigenstate, problem_)
                # print(offset, samples)
                # samples = [(res[0], problem_.objective.sense.value * (res[1] + offset), res[2])
                #    for res in samples]
                raw_samples.sort(key=lambda x: problem_.objective.sense.value * x.fval)
                x = raw_samples[0].x
                fval = raw_samples[0].fval

        # if Hamiltonian is empty, then the objective function is constant to the offset
        else:
            x = np.zeros(problem_.get_num_binary_vars())
            fval = offset
            raw_samples = [SolutionSample(x, fval, 1.0, OptimizationResultStatus.SUCCESS)]

        if fval is None or x is None:
            # if not function value is given, then something went wrong, e.g., a
            # NumPyMinimumEigensolver has been configured with an infeasible filter criterion.
            return MinimumEigenOptimizationResult(x=None, fval=None,
                                                  variables=problem.variables,
                                                  status=OptimizationResultStatus.FAILURE,
                                                  samples=None, raw_samples=None,
                                                  min_eigen_solver_result=eigen_result)
        # translate result back to integers
        samples = self._interpret_samples(problem, raw_samples)
        return cast(MinimumEigenOptimizationResult,
                    self._interpret(x=x, converters=self._converters, problem=problem,
                                    result_class=MinimumEigenOptimizationResult,
                                    samples=samples, raw_samples=raw_samples,
                                    min_eigen_solver_result=eigen_result))

    def _interpret_samples(self, problem: QuadraticProgram, raw_samples: List[SolutionSample]) \
            -> List[SolutionSample]:
        prob = {}  # type: dict
        array = {}
        for sample in raw_samples:
            x = sample.x
            for converter in self._converters[::-1]:
                x = converter.interpret(x)
            key = tuple(x)
            prob[key] = prob.get(key, 0.0) + sample.probability
            array[key] = x

        samples = []
        for key, x in array.items():
            probability = prob[key]
            fval = problem.objective.evaluate(x)
            status = self._get_feasibility_status(problem, x)
            samples.append(SolutionSample(x, fval, probability, status))

        return sorted(samples,
                      key=lambda v: (v.status.value, problem.objective.sense.value * v.fval))


def _eigenvector_to_solutions(eigenvector: Union[dict, np.ndarray, StateFn],
                              qubo: QuadraticProgram,
                              min_probability: float = 1e-6,
                              ) -> List[SolutionSample]:
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
        eigenvector = {bitstr: val ** 2 for (bitstr, val) in eigenvector.primitive.items()}
    elif isinstance(eigenvector, StateFn):
        eigenvector = eigenvector.to_matrix()

    def generate_solution(bitstr, qubo, probability):
        x = np.fromiter(list(bitstr), dtype=int)
        fval = qubo.objective.evaluate(x)
        return SolutionSample(x=x, fval=fval, probability=probability,
                              status=OptimizationResultStatus.SUCCESS)

    solutions = []
    if isinstance(eigenvector, dict):
        all_counts = sum(eigenvector.values())
        # iterate over all samples
        for bitstr, count in eigenvector.items():
            sampling_probability = count / all_counts
            # add the bitstring, if the sampling probability exceeds the threshold
            if sampling_probability >= min_probability:
                solutions.append(generate_solution(bitstr, qubo, sampling_probability))

    elif isinstance(eigenvector, np.ndarray):
        num_qubits = int(np.log2(eigenvector.size))
        probabilities = np.abs(eigenvector * eigenvector.conj())

        # iterate over all states and their sampling probabilities
        for i, sampling_probability in enumerate(probabilities):
            # add the i-th state if the sampling probability exceeds the threshold
            if sampling_probability >= min_probability:
                bitstr = '{:b}'.format(i).rjust(num_qubits, '0')[::-1]
                solutions.append(generate_solution(bitstr, qubo, sampling_probability))

    else:
        raise TypeError('Unsupported format of eigenvector. Provide a dict or numpy.ndarray.')

    return solutions
