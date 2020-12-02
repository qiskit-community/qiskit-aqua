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

"""GroverOptimizer module"""

import logging
import math
from copy import deepcopy
from typing import Optional, Dict, Union, List, cast

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms.amplitude_amplifiers.grover import Grover
from qiskit.circuit.library import QuadraticForm
from qiskit.providers import Backend, BaseBackend

from .optimization_algorithm import (OptimizationResultStatus, OptimizationAlgorithm,
                                     OptimizationResult)
from ..converters.quadratic_program_to_qubo import QuadraticProgramToQubo, QuadraticProgramConverter
from ..problems import Variable
from ..problems.quadratic_program import QuadraticProgram

logger = logging.getLogger(__name__)


class GroverOptimizer(OptimizationAlgorithm):
    """Uses Grover Adaptive Search (GAS) to find the minimum of a QUBO function."""

    def __init__(self, num_value_qubits: int, num_iterations: int = 3,
                 quantum_instance: Optional[Union[BaseBackend, Backend, QuantumInstance]] = None,
                 converters: Optional[Union[QuadraticProgramConverter,
                                            List[QuadraticProgramConverter]]] = None,
                 penalty: Optional[float] = None) -> None:
        """
        Args:
            num_value_qubits: The number of value qubits.
            num_iterations: The number of iterations the algorithm will search with
                no improvement.
            quantum_instance: Instance of selected backend, defaults to Aer's statevector simulator.
            converters: The converters to use for converting a problem into a different form.
                By default, when None is specified, an internally created instance of
                :class:`~qiskit.optimization.converters.QuadraticProgramToQubo` will be used.
            penalty: The penalty factor used in the default
                :class:`~qiskit.optimization.converters.QuadraticProgramToQubo` converter

        Raises:
            TypeError: When there one of converters is an invalid type.
        """
        self._num_value_qubits = num_value_qubits
        self._num_key_qubits = None
        self._n_iterations = num_iterations
        self._quantum_instance = None

        if quantum_instance is not None:
            self.quantum_instance = quantum_instance

        self._converters = self._prepare_converters(converters, penalty)

    @property
    def quantum_instance(self) -> QuantumInstance:
        """The quantum instance to run the circuits.

        Returns:
            The quantum instance used in the algorithm.
        """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[Backend,
                                                       BaseBackend, QuantumInstance]) -> None:
        """Set the quantum instance used to run the circuits.

        Args:
            quantum_instance: The quantum instance to be used in the algorithm.
        """
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            self._quantum_instance = QuantumInstance(quantum_instance)
        else:
            self._quantum_instance = quantum_instance

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

    def _get_a_operator(self, qr_key_value, problem):
        quadratic = problem.objective.quadratic.to_array()
        linear = problem.objective.linear.to_array()
        offset = problem.objective.constant

        # Get circuit requirements from input.
        quadratic_form = QuadraticForm(self._num_value_qubits, quadratic, linear, offset,
                                       little_endian=False)

        a_operator = QuantumCircuit(qr_key_value)
        a_operator.h(list(range(self._num_key_qubits)))
        a_operator.compose(quadratic_form, inplace=True)
        return a_operator

    def _get_oracle(self, qr_key_value):
        # Build negative value oracle O.
        if qr_key_value is None:
            qr_key_value = QuantumRegister(self._num_key_qubits + self._num_value_qubits)

        oracle_bit = QuantumRegister(1, "oracle")
        oracle = QuantumCircuit(qr_key_value, oracle_bit)
        oracle.z(self._num_key_qubits)  # recognize negative values.

        def is_good_state(self, measurement):
            """Check whether ``measurement`` is a good state or not."""
            value = measurement[self._num_key_qubits:self._num_key_qubits + self._num_value_qubits]
            return value[0] == '1'

        return oracle, is_good_state

    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """Tries to solves the given problem using the grover optimizer.

        Runs the optimizer to try to solve the optimization problem. If the problem cannot be,
        converted to a QUBO, this optimizer raises an exception due to incompatibility.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            AttributeError: If the quantum instance has not been set.
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """
        if self.quantum_instance is None:
            raise AttributeError('The quantum instance or backend has not been set.')

        self._verify_compatibility(problem)

        # convert problem to QUBO
        problem_ = self._convert(problem, self._converters)
        problem_init = deepcopy(problem_)

        # convert to minimization problem
        sense = problem_.objective.sense
        if sense == problem_.objective.Sense.MAXIMIZE:
            problem_.objective.sense = problem_.objective.Sense.MINIMIZE
            problem_.objective.constant = -problem_.objective.constant
            for i, val in problem_.objective.linear.to_dict().items():
                problem_.objective.linear[i] = -val
            for (i, j), val in problem_.objective.quadratic.to_dict().items():
                problem_.objective.quadratic[i, j] = -val
        self._num_key_qubits = len(problem_.objective.linear.to_array())  # type: ignore

        # Variables for tracking the optimum.
        optimum_found = False
        optimum_key = math.inf
        optimum_value = math.inf
        threshold = 0
        n_key = len(problem_.variables)
        n_value = self._num_value_qubits

        # Variables for tracking the solutions encountered.
        num_solutions = 2 ** n_key
        keys_measured = []

        # Variables for result object.
        operation_count = {}
        iteration = 0

        # Variables for stopping if we've hit the rotation max.
        rotations = 0
        max_rotations = int(np.ceil(100 * np.pi / 4))

        # Initialize oracle helper object.
        qr_key_value = QuantumRegister(self._num_key_qubits + self._num_value_qubits)
        orig_constant = problem_.objective.constant
        measurement = not self.quantum_instance.is_statevector
        oracle, is_good_state = self._get_oracle(qr_key_value)

        while not optimum_found:
            m = 1
            improvement_found = False

            # Get oracle O and the state preparation operator A for the current threshold.
            problem_.objective.constant = orig_constant - threshold
            a_operator = self._get_a_operator(qr_key_value, problem_)

            # Iterate until we measure a negative.
            loops_with_no_improvement = 0
            while not improvement_found:
                # Determine the number of rotations.
                loops_with_no_improvement += 1
                rotation_count = int(np.ceil(aqua_globals.random.uniform(0, m - 1)))
                rotations += rotation_count
                # Apply Grover's Algorithm to find values below the threshold.
                # TODO: Utilize Grover's incremental feature - requires changes to Grover.
                grover = Grover(oracle,
                                state_preparation=a_operator,
                                good_state=is_good_state)
                circuit = grover.construct_circuit(rotation_count, measurement=measurement)

                # Get the next outcome.
                outcome = self._measure(circuit)
                k = int(outcome[0:n_key], 2)
                v = outcome[n_key:n_key + n_value]
                int_v = self._bin_to_int(v, n_value) + threshold
                logger.info('Outcome: %s', outcome)
                logger.info('Value Q(x): %s', int_v)

                # If the value is an improvement, we update the iteration parameters (e.g. oracle).
                if int_v < optimum_value:
                    optimum_key = k
                    optimum_value = int_v
                    logger.info('Current Optimum Key: %s', optimum_key)
                    logger.info('Current Optimum Value: %s', optimum_value)
                    improvement_found = True
                    threshold = optimum_value
                else:
                    # Using Durr and Hoyer method, increase m.
                    m = int(np.ceil(min(m * 8 / 7, 2 ** (n_key / 2))))
                    logger.info('No Improvement. M: %s', m)

                    # Check if we've already seen this value.
                    if k not in keys_measured:
                        keys_measured.append(k)

                    # Assume the optimal if any of the stop parameters are true.
                    if loops_with_no_improvement >= self._n_iterations or \
                            len(keys_measured) == num_solutions or rotations >= max_rotations:
                        improvement_found = True
                        optimum_found = True

                # Track the operation count.
                operations = circuit.count_ops()
                operation_count[iteration] = operations
                iteration += 1
                logger.info('Operation Count: %s\n', operations)

        # If the constant is 0 and we didn't find a negative, the answer is likely 0.
        if optimum_value >= 0 and orig_constant == 0:
            optimum_key = 0

        opt_x = np.array([1 if s == '1' else 0 for s in ('{0:%sb}' % n_key).format(optimum_key)])

        # Compute function value
        fval = problem_init.objective.evaluate(opt_x)

        # cast binaries back to integers
        return cast(GroverOptimizationResult,
                    self._interpret(x=opt_x, converters=self._converters, problem=problem,
                                    result_class=GroverOptimizationResult,
                                    operation_counts=operation_count, n_input_qubits=n_key,
                                    n_output_qubits=n_value, intermediate_fval=fval,
                                    threshold=threshold))

    def _measure(self, circuit: QuantumCircuit) -> str:
        """Get probabilities from the given backend, and picks a random outcome."""
        probs = self._get_probs(circuit)
        freq = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        # Pick a random outcome.
        freq[-1] = (freq[-1][0], 1.0 - sum(x[1] for x in freq[0:len(freq) - 1]))
        idx = aqua_globals.random.choice(len(freq), 1, p=[x[1] for x in freq])[0]
        logger.info('Frequencies: %s', freq)

        return freq[idx][0]

    def _get_probs(self, qc: QuantumCircuit) -> Dict[str, float]:
        """Gets probabilities from a given backend."""
        # Execute job and filter results.
        result = self.quantum_instance.execute(qc)
        if self.quantum_instance.is_statevector:
            state = np.round(result.get_statevector(qc), 5)
            keys = [bin(i)[2::].rjust(int(np.log2(len(state))), '0')[::-1]
                    for i in range(0, len(state))]
            probs = [np.round(abs(a) * abs(a), 5) for a in state]
            hist = dict(zip(keys, probs))
        else:
            state = result.get_counts(qc)
            shots = self.quantum_instance.run_config.shots
            hist = {}
            for key in state:
                hist[key[::-1]] = state[key] / shots
        hist = dict(filter(lambda p: p[1] > 0, hist.items()))
        return hist

    @staticmethod
    def _bin_to_int(v: str, num_value_bits: int) -> int:
        """Converts a binary string of n bits using two's complement to an integer."""
        if v.startswith("1"):
            int_v = int(v, 2) - 2 ** num_value_bits
        else:
            int_v = int(v, 2)

        return int_v


class GroverOptimizationResult(OptimizationResult):
    """A result object for Grover Optimization methods."""

    def __init__(self, x: Union[List[float], np.ndarray], fval: float, variables: List[Variable],
                 operation_counts: Dict[int, Dict[str, int]], n_input_qubits: int,
                 n_output_qubits: int, intermediate_fval: float, threshold: float,
                 status: OptimizationResultStatus) -> None:
        """
        Constructs a result object with the specific Grover properties.

        Args:
            x: The solution of the problem
            fval: The value of the objective function of the solution
            variables: A list of variables defined in the problem
            operation_counts: The counts of each operation performed per iteration.
            n_input_qubits: The number of qubits used to represent the input.
            n_output_qubits: The number of qubits used to represent the output.
            intermediate_fval: The intermediate value of the objective function of the solution,
                that is expected to be identical with ``fval``.
            threshold: The threshold of Grover algorithm.
            status: the termination status of the optimization algorithm.
        """
        super().__init__(x, fval, variables, status, None)
        self._operation_counts = operation_counts
        self._n_input_qubits = n_input_qubits
        self._n_output_qubits = n_output_qubits
        self._intermediate_fval = intermediate_fval
        self._threshold = threshold

    @property
    def operation_counts(self) -> Dict[int, Dict[str, int]]:
        """Get the operation counts.

        Returns:
            The counts of each operation performed per iteration.
        """
        return self._operation_counts

    @property
    def n_input_qubits(self) -> int:
        """Getter of n_input_qubits

        Returns:
            The number of qubits used to represent the input.
        """
        return self._n_input_qubits

    @property
    def n_output_qubits(self) -> int:
        """Getter of n_output_qubits

        Returns:
            The number of qubits used to represent the output.
        """
        return self._n_output_qubits

    @property
    def intermediate_fval(self) -> float:
        """Getter of the intermediate fval

        Returns:
            The intermediate value of fval before interpret.
        """
        return self._intermediate_fval

    @property
    def threshold(self) -> float:
        """Getter of the threshold of Grover algorithm.

        Returns:
            The threshold of Grover algorithm.
        """
        return self._threshold
