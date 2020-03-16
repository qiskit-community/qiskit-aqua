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

"""GroverMinimumFinder module"""

import logging
from typing import Optional, Dict
import random
import math
import numpy as np
from qiskit.optimization.algorithms import OptimizationAlgorithm
from qiskit.optimization.problems import OptimizationProblem
from qiskit.optimization.converters import (IntegerToBinaryConverter,
                                            PenalizeLinearEqualityConstraints,
                                            OptimizationProblemToNegativeValueOracle)
from qiskit.optimization.results import GroverOptimizationResults
from qiskit.optimization.results import OptimizationResult
from qiskit.optimization.utils import QiskitOptimizationError
from qiskit.optimization.util import get_qubo_solutions
from qiskit.aqua.algorithms.amplitude_amplifiers.grover import Grover
from qiskit import Aer, execute, QuantumCircuit
from qiskit.providers import BaseBackend


class GroverMinimumFinder(OptimizationAlgorithm):

    """Uses Grover Adaptive Search (GAS) to find the minimum of a QUBO function."""

    def __init__(self, num_iterations: int = 3, backend: Optional[BaseBackend] = None) -> None:
        """
        Args:
            num_iterations: The number of iterations the algorithm will search with
                no improvement.
            backend: Instance of selected backend, defaults to Aer's statevector simulator.
        """
        self._n_iterations = num_iterations
        if backend is None:
            self._backend = Aer.get_backend('statevector_simulator')
        else:
            self._backend = backend
        self._logger = logging.getLogger(__name__)

    def is_compatible(self, problem: OptimizationProblem) -> Optional[str]:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem contains only
        binary and integer variables as well as linear equality constraints, and otherwise,
        returns a message explaining the incompatibility.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns ``None`` if the problem is compatible and else a string with the error message.
        """

        # initialize message
        msg = ''

        # check whether there are incompatible variable types
        if problem.variables.get_num_continuous() > 0:
            msg += 'Continuous variables are not supported! '
        if problem.variables.get_num_semicontinuous() > 0:
            msg += 'Semi-continuous variables are not supported! '
        # if problem.variables.get_num_integer() > 0:
        #     # TODO: to be removed once integer to binary mapping is introduced
        #     msg += 'Integer variables are not supported! '
        if problem.variables.get_num_semiinteger() > 0:
            # TODO: to be removed once semi-integer to binary mapping is introduced
            msg += 'Semi-integer variables are not supported! '

        # check whether there are incompatible constraint types
        if not all([sense == 'E' for sense in problem.linear_constraints.get_senses()]):
            msg += 'Only linear equality constraints are supported.'

        # TODO: check for quadratic constraints

        # if an error occurred, return error message, otherwise, return None
        if len(msg) > 0:
            return msg.strip()
        else:
            return None

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
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

        # analyze compatibility of problem
        msg = self.is_compatible(problem)
        if msg is not None:
            raise QiskitOptimizationError('Incompatible problem: %s' % msg)

        # map integer variables to binary variables
        int_to_bin_converter = IntegerToBinaryConverter()
        problem_ = int_to_bin_converter.encode(problem)

        # penalize linear equality constraints with only binary variables
        penalty = 2  # TODO
        # if self._penalty is None:
        #     # TODO: should be derived from problem
        #     penalty = 1e5
        # else:
        #     penalty = self._penalty
        lin_eq_converter = PenalizeLinearEqualityConstraints()
        problem_ = lin_eq_converter.encode(problem_, penalty_factor=penalty)

        # TODO: How to get from Optimization Problem?
        num_output_qubits = 6

        # Variables for tracking the optimum.
        optimum_found = False
        optimum_key = math.inf
        optimum_value = math.inf
        threshold = 0
        n_key = problem_.variables.get_num()
        n_value = num_output_qubits

        # Variables for tracking the solutions encountered.
        num_solutions = 2**n_key
        keys_measured = []

        # Variables for result object.
        func_dict = {}
        operation_count = {}
        iteration = 0

        # Variables for stopping if we've hit the rotation max.
        rotations = 0
        max_rotations = int(np.ceil(100*np.pi/4))

        # Initialize oracle helper object.
        orig_constant = problem_.objective.get_offset()
        opt_prob_converter = OptimizationProblemToNegativeValueOracle(n_value,
                                                                      backend=self._backend)

        while not optimum_found:
            m = 1
            improvement_found = False

            # Get oracle O and the state preparation operator A for the current threshold.
            problem_.objective.set_offset(orig_constant - threshold)
            a_operator, oracle, func_dict = opt_prob_converter.encode(problem_)

            # Iterate until we measure a negative.
            loops_with_no_improvement = 0
            while not improvement_found:
                # Determine the number of rotations.
                loops_with_no_improvement += 1
                rotation_count = int(np.ceil(random.uniform(0, m-1)))
                rotations += rotation_count

                # Apply Grover's Algorithm to find values below the threshold.
                if rotation_count > 0:
                    grover = Grover(oracle, init_state=a_operator, num_iterations=rotation_count)
                    circuit = grover.construct_circuit(
                        measurement=self._backend.name() != "statevector_simulator")
                else:
                    circuit = a_operator._circuit

                # Get the next outcome.
                outcome = self._measure(circuit, n_key, n_value, self._backend)
                k = int(outcome[0:n_key], 2)
                v = outcome[n_key:n_key + n_value]

                # Convert the binary string to integer.
                int_v = self._bin_to_int(v, n_value) + threshold
                v = self._twos_complement(int_v, n_value)

                self._logger.info('Iterations: {}'.format(rotation_count))
                self._logger.info('Outcome: {}'.format(outcome))
                self._logger.info('Value: {} = {}'.format(v, int_v))

                # If the value is an improvement, we update the iteration parameters (e.g. oracle).
                if int_v < optimum_value:
                    optimum_key = k
                    optimum_value = int_v
                    self._logger.info('Current Optimum Key: {}'.format(optimum_key))
                    self._logger.info('Current Optimum Value: {}'.format(optimum_value))
                    if v.startswith("1"):
                        improvement_found = True
                        threshold = optimum_value
                else:
                    # No better number after the max number of iterations, so we assume the optimal.
                    if loops_with_no_improvement >= self._n_iterations:
                        improvement_found = True
                        optimum_found = True

                    # Using Durr and Hoyer method, increase m.
                    # TODO: Give option for a rotation schedule, or for different lambda's.
                    m = int(np.ceil(min(m * 8/7, 2**(n_key / 2))))
                    self._logger.info('No Improvement. M: {}'.format(m))

                # Check if we've already seen this value.
                if k not in keys_measured:
                    keys_measured.append(k)

                # Stop if we've seen all the keys or hit the rotation max.
                if len(keys_measured) == num_solutions or rotations >= max_rotations:
                    improvement_found = True
                    optimum_found = True

                # Track the operation count.
                operations = circuit.count_ops()
                operation_count[iteration] = operations
                iteration += 1
                self._logger.info('Operation Count: {}\n'.format(operations))

        # Get original key and value pairs.
        func_dict[-1] = orig_constant
        solutions = get_qubo_solutions(func_dict, n_key)

        # If the constant is 0 and we didn't find a negative, the answer is likely 0.
        if optimum_value >= 0 and orig_constant == 0:
            optimum_key = 0
        opt_x = [1 if s == '1' else 0 for s in ('{0:%sb}' % n_key).format(optimum_key)]

        # Build the results object.
        grover_results = GroverOptimizationResults(operation_count, rotations, n_key, n_value,
                                                   func_dict)
        result = OptimizationResult(x=opt_x, fval=solutions[optimum_key],
                                    results=grover_results)

        # cast binaries back to integers
        result = int_to_bin_converter.decode(result)

        return result

    def _measure(self, circuit: QuantumCircuit, n_key: int, n_value: int, backend: BaseBackend,
                 shots: Optional[int] = 1024) -> str:
        """Get probabilities from the given backend, and picks a random outcome."""
        probs = self._get_probs(n_key, n_value, circuit, backend, shots)
        freq = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        # Pick a random outcome.
        freq[len(freq)-1] = (freq[len(freq)-1][0], 1 - sum([x[1] for x in freq[0:len(freq)-1]]))
        idx = np.random.choice(len(freq), 1, p=[x[1] for x in freq])[0]
        self._logger.info('Frequencies: {}'.format(freq))

        return freq[idx][0]

    @staticmethod
    def _get_probs(n_key: int, n_value: int, qc: QuantumCircuit, backend: BaseBackend,
                   shots: int) -> Dict[str, float]:
        """Gets probabilities from a given backend."""
        # Execute job and filter results.
        job = execute(qc, backend=backend, shots=shots)
        result = job.result()
        if backend.name() == 'statevector_simulator':
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
