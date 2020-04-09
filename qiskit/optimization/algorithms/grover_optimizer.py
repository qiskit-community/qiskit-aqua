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

"""GroverOptimizer module"""

import logging
from typing import Optional, Dict, Union
import math
import numpy as np
from qiskit.aqua import QuantumInstance
from qiskit.optimization.algorithms import OptimizationAlgorithm
from qiskit.optimization.problems import QuadraticProgram
from qiskit.optimization.converters import (QuadraticProgramToQubo,
                                            QuadraticProgramToNegativeValueOracle)
from qiskit.optimization.results import GroverOptimizationResults
from qiskit.optimization.results import OptimizationResult
from qiskit.optimization.util import get_qubo_solutions
from qiskit.aqua.algorithms.amplitude_amplifiers.grover import Grover
from qiskit import Aer, QuantumCircuit
from qiskit.providers import BaseBackend


logger = logging.getLogger(__name__)


class GroverOptimizer(OptimizationAlgorithm):
    """Uses Grover Adaptive Search (GAS) to find the minimum of a QUBO function."""

    def __init__(self, num_value_qubits: int, num_iterations: int = 3,
                 quantum_instance: Optional[Union[BaseBackend, QuantumInstance]] = None) -> None:
        """
        Args:
            num_value_qubits: The number of value qubits.
            num_iterations: The number of iterations the algorithm will search with
                no improvement.
            quantum_instance: Instance of selected backend, defaults to Aer's statevector simulator.
        """
        self._num_value_qubits = num_value_qubits
        self._n_iterations = num_iterations
        if quantum_instance is None or isinstance(quantum_instance, BaseBackend):
            backend = quantum_instance or Aer.get_backend('statevector_simulator')
            quantum_instance = QuantumInstance(backend)
        self._quantum_instance = quantum_instance

    def is_compatible(self, problem: QuadraticProgram) -> Optional[str]:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem can be converted
        to a QUBO, and otherwise, returns a message explaining the incompatibility.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns ``None`` if the problem is compatible and else a string with the error message.
        """
        return QuadraticProgramToQubo.is_compatible(problem)

    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem. If problem is not convex,
        this optimizer may raise an exception due to incompatibility, depending on the settings.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """

        # convert problem to QUBO
        qubo_converter = QuadraticProgramToQubo()
        problem_ = qubo_converter.encode(problem)

        # Variables for tracking the optimum.
        optimum_found = False
        optimum_key = math.inf
        optimum_value = math.inf
        threshold = 0
        n_key = problem_.variables.get_num()
        n_value = self._num_value_qubits

        # Variables for tracking the solutions encountered.
        num_solutions = 2**n_key
        keys_measured = []

        # Variables for result object.
        func_dict = {}
        operation_count = {}
        iteration = 0

        # Initialize oracle helper object.
        orig_constant = problem_.objective.get_offset()
        measurement = not self._quantum_instance.is_statevector
        opt_prob_converter = QuadraticProgramToNegativeValueOracle(n_value,
                                                                   measurement)

        loops_with_no_improvement = 0
        while not optimum_found:
            # Get oracle O and the state preparation operator A for the current threshold.
            problem_.objective.set_offset(orig_constant - threshold)
            a_operator, oracle, func_dict = opt_prob_converter.encode(problem_)

            # Apply Grover's Algorithm to find values below the threshold.
            grover = Grover(oracle, init_state=a_operator, incremental=True)
            circuit = grover.construct_circuit(measurement=self._quantum_instance.is_statevector)

            # Get the next outcome.
            outcome = self._measure(circuit, n_key, n_value)
            k = int(outcome[0:n_key], 2)
            v = outcome[n_key:n_key + n_value]
            int_v = self._bin_to_int(v, n_value) + threshold
            v = self._twos_complement(int_v, n_value)
            logger.info('Outcome: %s', outcome)
            logger.info('Value: %s = %s', v, int_v)

            # If the value is an improvement, we update the iteration parameters (e.g. oracle).
            if int_v < optimum_value:
                optimum_key = k
                optimum_value = int_v
                logger.info('Current Optimum Key: %s', optimum_key)
                logger.info('Current Optimum Value: %s', optimum_value)
                if v.startswith('1'):
                    threshold = optimum_value
            else:
                # No better number after the max number of iterations, so we assume the optimal.
                if loops_with_no_improvement >= self._n_iterations:
                    optimum_found = True

            # Check if we've already seen this value.
            if k not in keys_measured:
                keys_measured.append(k)

            # Stop if we've seen all the keys.
            if len(keys_measured) == num_solutions:
                optimum_found = True

            # Track the operation count.
            operations = circuit.count_ops()
            operation_count[iteration] = operations
            iteration += 1
            logger.info('Operation Count: %s\n', operations)

        # Get original key and value pairs.
        func_dict[-1] = orig_constant
        solutions = get_qubo_solutions(func_dict, n_key)

        # If the constant is 0 and we didn't find a negative, the answer is likely 0.
        if optimum_value >= 0 and orig_constant == 0:
            optimum_key = 0
        opt_x = [1 if s == '1' else 0 for s in ('{0:%sb}' % n_key).format(optimum_key)]

        # Build the results object.
        grover_results = GroverOptimizationResults(operation_count, n_key, n_value, func_dict)
        result = OptimizationResult(x=opt_x, fval=solutions[optimum_key],
                                    results={"grover_results": grover_results,
                                             "qubo_converter": qubo_converter})

        # cast binaries back to integers
        result = qubo_converter.decode(result)

        return result

    def _measure(self, circuit: QuantumCircuit, n_key: int, n_value: int) -> str:
        """Get probabilities from the given backend, and picks a random outcome."""
        probs = self._get_probs(n_key, n_value, circuit)
        freq = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        # Pick a random outcome.
        freq[len(freq)-1] = (freq[len(freq)-1][0], 1 - sum([x[1] for x in freq[0:len(freq)-1]]))
        idx = np.random.choice(len(freq), 1, p=[x[1] for x in freq])[0]
        logger.info('Frequencies: %s', freq)

        return freq[idx][0]

    def _get_probs(self, n_key: int, n_value: int, qc: QuantumCircuit) -> Dict[str, float]:
        """Gets probabilities from a given backend."""
        # Execute job and filter results.
        result = self._quantum_instance.execute(qc)
        if self._quantum_instance.is_statevector:
            state = np.round(result.get_statevector(qc), 5)
            keys = [bin(i)[2::].rjust(int(np.log2(len(state))), '0')[::-1]
                    for i in range(0, len(state))]
            probs = [np.round(abs(a)*abs(a), 5) for a in state]
            f_hist = dict(zip(keys, probs))
            hist = {}
            for key in f_hist:
                new_key = key[:n_key] + key[n_key:n_key+n_value][::-1] + key[n_key+n_value:]
                hist[new_key] = f_hist[key]
        else:
            state = result.get_counts(qc)
            shots = self._quantum_instance.run_config.shots
            hist = {}
            for key in state:
                hist[key[:n_key] + key[n_key:n_key+n_value][::-1] + key[n_key+n_value:]] = \
                    state[key] / shots
        hist = dict(filter(lambda p: p[1] > 0, hist.items()))

        return hist

    @staticmethod
    def _twos_complement(v: int, n_bits: int) -> str:
        """Converts an integer into a binary string of n bits using two's complement."""
        assert -2**n_bits <= v < 2**n_bits

        if v < 0:
            v += 2**n_bits
            bin_v = bin(v)[2:]
        else:
            format_string = '{0:0'+str(n_bits)+'b}'
            bin_v = format_string.format(v)

        return bin_v

    @staticmethod
    def _bin_to_int(v: str, num_value_bits: int) -> int:
        """Converts a binary string of n bits using two's complement to an integer."""
        if v.startswith("1"):
            int_v = int(v, 2) - 2 ** num_value_bits
        else:
            int_v = int(v, 2)

        return int_v
