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

# todo: update documentation
"""Implementation of the warm start QAOA optimizer."""

import copy
from abc import ABC
from typing import Optional, List, Union, Dict, Tuple, cast

import numpy as np
from qiskit import QuantumCircuit
from qiskit.aqua.algorithms import QAOA
from qiskit.circuit import Parameter
from qiskit.optimization import QuadraticProgram, QiskitOptimizationError
from qiskit.optimization.algorithms import OptimizationAlgorithm, MinimumEigenOptimizer, \
    MinimumEigenOptimizationResult
from qiskit.optimization.converters import QuadraticProgramConverter
from qiskit.optimization.problems import VarType


class BaseAggregator(ABC):
    """A base abstract class for aggregates results"""

    def aggregate(self, results: List[MinimumEigenOptimizationResult]) \
            -> List[Tuple[str, float, float]]:
        """
        Aggregates the results.

        Args:
            results: List of result objects that need to be combined.

        Returns:
             Aggregated samples.
        """
        raise NotImplementedError


class MeanAggregator(BaseAggregator):
    """Aggregates the results by averaging the probability of each sample."""

    def aggregate(self, results: List[MinimumEigenOptimizationResult]) \
            -> List[Tuple[str, float, float]]:
        """
        Args:
            results: List of result objects that need to be combined.

        Returns:
             Aggregated samples by averaging them.
        """

        # Use a dict for fast solution look-up
        # Key: sample code, value: tuple of fval, probability
        dict_samples: Dict[str, Tuple[float, float]] = {}

        # Sum up all the probabilities in the results
        for result in results:
            for sample in result.samples:
                state, fval, prob = sample[0], sample[1], sample[2]
                if state in dict_samples:
                    dict_samples[state] = (fval, dict_samples[state][1] + prob)
                else:
                    dict_samples[state] = (fval, prob)

        # Divide by the number of results to normalize
        new_samples = []
        num_results = len(results)
        for state in dict_samples:
            sample = (state, dict_samples[state][0], dict_samples[state][1] / num_results)
            new_samples.append(sample)

        # new_samples.sort(key=lambda sample: problem.objective.sense.value * sample[1])
        # x = [float(e) for e in new_samples[0][0]]

        return new_samples


class WarmStartQAOACircuitFactory:
    """
    A factory that produces quantum circuits for the QAOA implementation. The methods of this
    factory can be overridden to modify behavior of QAOA. This implementation generates quantum
    circuits for initial state and mixer to warm start QAOA.
    """

    def __init__(self, epsilon: float) -> None:
        """
        Args:
            epsilon: the regularization parameter that changes the initial variables according to
                xi = epsilon if xi < epsilon
                xi = 1-epsilon if xi > epsilon.
                The regularization parameter epsilon should be between 0 and 0.5. When it
                is 0.5 then warm start corresponds to standard QAOA.
        Raises:
            AquaError: if ``epsilon`` is not specified for the warm start QAOA or value is not in
                the range [0, 0.5].
        """
        self._epsilon = epsilon

    def create_initial_variables(self, solution: List[float]) -> List[float]:
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

    def create_initial_state(self, initial_variables: List[float]) -> QuantumCircuit:
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

    def create_mixer(self, initial_variables: List[float]) -> QuantumCircuit:
        """
        Creates an evolved mixer circuit as Ry(theta)Rz(-2beta)Ry(-theta).

        Args:
            initial_variables: Already created initial variables.

        Returns:
            A quantum circuit to be used as a mixer in QAOA.
        """
        circuit = QuantumCircuit(len(initial_variables))
        beta = Parameter("beta")

        for index, relaxed_value in enumerate(initial_variables):
            theta = 2 * np.arcsin(np.sqrt(relaxed_value))

            circuit.ry(-theta, index)
            circuit.rz(-2.0 * beta, index)
            circuit.ry(theta, index)

        return circuit


class WarmStartQAOAOptimizer(MinimumEigenOptimizer):
    # todo: update documentation
    """A meta-algorithm that uses a pre-solver to solve a relaxed version of the problem.
    Users must implement their own pre solvers by inheriting from the base class."""

    def __init__(self,
                 pre_solver: OptimizationAlgorithm,
                 relax_for_pre_solver: bool,
                 qaoa: QAOA,
                 epsilon: Optional[float] = None,
                 num_initial_solutions: int = 1,
                 circuit_factory: Optional[WarmStartQAOACircuitFactory] = None,
                 aggregator: Optional[BaseAggregator] = None,
                 penalty: Optional[float] = None,
                 converters: Optional[Union[QuadraticProgramConverter,
                                            List[QuadraticProgramConverter]]] = None
                 ) -> None:
        """ Initializes the warm start minimum eigen optimizer. For correct initialization either
            ``epsilon`` or ``circuit_factory`` must be specified. If only ``epsilon`` is specified,
            then a default ``WarmStartQAOACircuitFactory`` is created. If only ``circuit_factory``
            is specified then this instance is used in the implementation. If both parameters are
            specified then the circuit factory is used and ``epsilon`` is no longer relevant.

        Args:
            pre_solver: An instance of an optimizer to solve the relaxed version of the problem.
            relax_for_pre_solver: True if the problem must be relaxed to the continuous case
                before passing it to the pre-solver.
            qaoa: A QAOA instance to be used in the computations.
            epsilon: the regularization parameter that changes the initial variables according to
                xi = epsilon if xi < epsilon
                xi = 1-epsilon if xi > epsilon.
                The regularization parameter epsilon should be between 0 and 0.5. When it
                is 0.5 then warm start corresponds to standard QAOA. If ``circuit_factory`` is
                specified then this parameter is ignored.
            num_initial_solutions: A number of relaxed (continuous) solutions to use.
            circuit_factory: An instance of the circuit factory to be used to create circuits for
                the initial state and mixer. If none is passed then a default one,
                ``WarmStartQAOACircuitFactory`` is created using ``epsilon`` parameter.
            aggregator: Class that aggregates different results. This is used if the pre-solver
                returns several initial states.
            penalty: The penalty factor to be used, or ``None`` for applying a default logic.
            converters: The converters to use for converting a problem into a different form.
                By default, when None is specified, an internally created instance of
                :class:`~qiskit.optimization.converters.QuadraticProgramToQubo` will be used.

        Raises:
            QiskitOptimizationError: if both ``epsilon`` and ``circuit_factory`` are not specified
                or ``epsilon`` value is not in the range [0, 0.5].
        """
        if epsilon is None and circuit_factory is None:
            raise QiskitOptimizationError(
                'Either epsilon or circuit_factory must be specified for the warm start QAOA')

        if epsilon is not None and (epsilon < 0. or epsilon > 0.5):
            raise QiskitOptimizationError(
                'Epsilon for warm-start QAOA needs to be between 0 and 0.5.')

        self._pre_solver = pre_solver
        self._relax_for_pre_solver = relax_for_pre_solver
        self._qaoa = qaoa
        self._epsilon = epsilon
        self._num_initial_solutions = num_initial_solutions

        if circuit_factory is None:
            circuit_factory = WarmStartQAOACircuitFactory(self._epsilon)
        self._circuit_factory = circuit_factory

        if num_initial_solutions > 1 and aggregator is None:
            aggregator = MeanAggregator()
        self._aggregator = aggregator

        super().__init__(qaoa, penalty, converters)

    def solve(self, problem: QuadraticProgram) -> MinimumEigenOptimizationResult:
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

        # convert problem to QUBO or another form if converters are specified
        converted_problem = self._convert(problem, self._converters)

        # if the pre-solver can't solve the problem then it should be relaxed.
        if self._relax_for_pre_solver:
            pre_solver_problem = self._relax_problem(converted_problem)
        else:
            pre_solver_problem = converted_problem

        opt_result = self._pre_solver.solve(pre_solver_problem)
        num_initial_solutions = min(self._num_initial_solutions, len(opt_result.explored_solutions))
        initial_solutions = opt_result.explored_solutions[:num_initial_solutions]

        # construct operator and offset
        operator, offset = converted_problem.to_ising()

        results = []  # type: List[MinimumEigenOptimizationResult]
        for initial_solution, _ in initial_solutions:
            # Set the solver using the result of the pre-solver.
            initial_variables = self._circuit_factory.create_initial_variables(initial_solution)
            self._qaoa.initial_state = self._circuit_factory.create_initial_state(initial_variables)
            self._qaoa.mixer = self._circuit_factory.create_mixer(initial_variables)

            # approximate ground state of operator using min eigen solver.
            results.append(self._solve_internal(operator, offset, converted_problem, problem))

        if len(results) == 1:
            # there's no need to call _interpret, it is done by MinimumEigenOptimizer
            return results[0]
        else:
            samples = self._aggregator.aggregate(results)

            samples.sort(key=lambda sample: problem.objective.sense.value * sample[1])
            x = [float(e) for e in samples[0][0]]

            # translate result back to the original variables
            return cast(MinimumEigenOptimizationResult,
                        self._interpret(x=x, converters=self._converters, problem=problem,
                                        result_class=MinimumEigenOptimizationResult,
                                        samples=samples))

    @staticmethod
    def _relax_problem(problem: QuadraticProgram) -> QuadraticProgram:
        """
        Change all variables to continuous.

        Args:
            problem: Problem to relax.

        Returns:
            A copy of the original problem where all variables are continuous.
        """
        relaxed_problem = copy.deepcopy(problem)
        for variable in relaxed_problem.variables:
            variable.vartype = VarType.CONTINUOUS

        return relaxed_problem
