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

# todo: update docstring
"""Implementation of the warm start QAOA optimizer."""

from abc import ABC
from typing import Optional, List, Union

import numpy as np
from qiskit.circuit import Parameter

from qiskit import QuantumCircuit
from qiskit.aqua import AquaError
from qiskit.aqua.algorithms import QAOA
from qiskit.optimization import QuadraticProgram, QiskitOptimizationError
from qiskit.optimization.algorithms import OptimizationAlgorithm, OptimizationResult, \
    MinimumEigenOptimizer, MinimumEigenOptimizationResult, OptimizationResultStatus
from qiskit.optimization.converters import QuadraticProgramToQubo, QuadraticProgramConverter


class BaseAggregator(ABC):
    """Aggregates results"""

    def aggregate(self, results: List[MinimumEigenOptimizationResult], problem: QuadraticProgram,
                  qubo_converter: QuadraticProgramToQubo) -> MinimumEigenOptimizationResult:
        """Aggregates the results."""
        raise NotImplementedError


class MeanAggregator(BaseAggregator):
    """Aggregates the results by averaging the probability of each sample."""

    def aggregate(self, results: List[MinimumEigenOptimizationResult],
                  problem: QuadraticProgram, qubo_converter: QuadraticProgramToQubo):
        """
        Args:
            results: List of result objects that need to be combined.
            problem:
            qubo_converter:

        Returns:
             An aggregated result.
        """

        # Use a dict for fast solution look-up
        dict_samples = {}
        n_results = len(results)

        # Sum up all the probabilities in the results
        for result in results:
            for sample in result.samples:
                if sample[0] in dict_samples:
                    dict_samples[sample[0]] = (sample[1], dict_samples[sample[0]][1] + sample[2])
                else:
                    dict_samples[sample[0]] = (sample[1], sample[2])

        # Divide by the number of results to normalize
        new_samples = []
        for state in dict_samples:
            sample = (state, dict_samples[state][0], dict_samples[state][1] / n_results)
            new_samples.append(sample)

        # todo: this snippet is also used in min eigen optimizer, consider making it as a function
        new_samples.sort(key=lambda sample: problem.objective.sense.value * sample[1])
        x = [float(e) for e in new_samples[0][0]]
        fval = new_samples[0][1]

        # todo: we have to call _interpret to convert to the original representation
        return MinimumEigenOptimizationResult(x, fval, variables=problem.variables,
                                              status=OptimizationResultStatus.SUCCESS,
                                              samples=new_samples)


class WarmStartQAOAOptimizer(MinimumEigenOptimizer):
    # todo: update docstring
    """A meta-algorithm that uses a pre-solver to solve a relaxed version of the problem.
    Users must implement their own pre solvers by inheriting from the base class."""

    def __init__(self,
                 pre_solver: OptimizationAlgorithm,
                 qaoa: QAOA,
                 epsilon: float,
                 aggregator: Optional[BaseAggregator] = None,
                 penalty: Optional[float] = None,
                 converters: Optional[Union[QuadraticProgramConverter,
                                            List[QuadraticProgramConverter]]] = None
                 ) -> None:
        """ Initializes the warm start minimum eigen optimizer.

        Args:
            pre_solver: An instance of an optimizer to solve the relaxed version of the problem.
            qaoa: A QAOA instance to be used in the computations.
            epsilon: the regularization parameter that changes the initial variables
                according to
                xi = epsilon if xi < epsilon
                xi = 1-epsilon if xi > epsilon.
                The regularization parameter epsilon should be between 0 and 0.5. When it
                is 0.5 then warm start corresponds to standard QAOA.
            penalty: The penalty factor to be used, or ``None`` for applying a default logic.
            aggregator: Class that aggregates different results. This is used if the pre-solver
                returns several initial states.

        Raises:
            AquaError: if ``epsilon`` is not specified for the warm start QAOA.
        """
        if epsilon is None:
            raise AquaError('Epsilon must be specified for the warm start QAOA')

        self._pre_solver = pre_solver
        self._qaoa = qaoa
        self._epsilon = epsilon
        self._aggregator = aggregator
        super().__init__(qaoa, penalty, converters)

    def get_compatibility_msg(self, problem: QuadraticProgram) -> Optional[str]:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem can be converted
        to a QUBO, and otherwise, returns a message explaining the incompatibility.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            A message describing the incompatibility.
        """
        return super().get_compatibility_msg(problem)

    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """Tries to solves the given problem using the optimizer.

        The pre-solver is run to warm-start the solver. Next, the optimizer is run
        to try to solve the optimization problem.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If problem not compatible.
        """

        msg = self.get_compatibility_msg(problem)
        if len(msg) > 0:
            raise QiskitOptimizationError('Incompatible problem: {}'.format(msg))

        # convert problem to QUBO
        # qubo_converter = QuadraticProgramToQubo(self._penalty)
        # qubo_problem = qubo_converter.convert(problem)
        qubo_problem = self._convert(problem, self._converters)

        # construct operator and offset
        operator, offset = qubo_problem.to_ising()
        # operator_converter = QuadraticProgramToIsing()
        # operator, offset = operator_converter.encode(qubo_problem)

        opt_result = self._pre_solver.solve(problem)
        # todo: accessing all solutions
        relaxed_solutions = opt_result.all_solutions

        results = []  # type: List[MinimumEigenOptimizationResult]
        for relaxed_solution, value in relaxed_solutions:
            # Set the solver using the result of the pre-solver.
            initial_variables = self._create_initial_variables(relaxed_solution)
            # todo: no such setter
            self._qaoa.initial_state = self._create_initial_state(initial_variables)
            # todo: no such setter
            self._qaoa.mixer = self._create_mixer(initial_variables)

            # approximate ground state of operator using min eigen solver.
            results.append(self._solve_internal(operator, offset, qubo_problem, problem))

        if len(results) == 1:
            return results[0]
        else:
            # todo: handle converters and _interpret
            return self._aggregator.aggregate(results, qubo_problem, None)

    def _create_initial_variables(self, solution: List[float]) -> List[float]:
        """
        Creates initial variable values to warm start QAOA.

        Args:
            solution: a solution obtained for the relaxed problem.

        Returns:
            A list of initial variables constructed from a relaxed solution.
        """
        initial_variables = []

        for variable in solution:
            if variable < self._epsilon:
                initial_variables.append(self._epsilon)
            elif variable > 1. - self._epsilon:
                initial_variables.append(1. - self._epsilon)
            else:
                initial_variables.append(variable)

        return initial_variables

    @staticmethod
    def _create_initial_state(initial_variables: List[float]) -> QuantumCircuit:
        """
        Creates an initial state quantum circuit to warm start QAOA.

        Args:
            initial_variables: Already created initial variables.

        Returns:
            A quantum circuit that represents initial state.
        """
        circuit = QuantumCircuit(len(initial_variables))

        for index, relaxed_value in enumerate(initial_variables):
            theta = 2 * np.arcsin(np.sqrt(relaxed_value))
            circuit.ry(theta, index)

        return circuit

    @staticmethod
    def _create_mixer(initial_variables: List[float]) -> QuantumCircuit:
        """
        Creates the evolved mixer circuit as Ry(theta)Rz(-2beta)Ry(-theta).

        Args:
            initial_variables: Already created initial variables.

        Returns:
            A quantum circuit to be used as a mixer in QAOA.
        """
        circuit = QuantumCircuit(len(initial_variables))
        beta = Parameter("beta")

        for index, relaxed_value in enumerate(initial_variables):
            theta = 2 * np.arcsin(np.sqrt(relaxed_value))

            circuit.ry(theta, index)
            circuit.rz(-2.0 * beta, index)
            circuit.ry(-theta, index)

        return circuit
