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

import random
import math
import numpy as np
from qiskit.optimization.converters import OptimizationProblemToNegativeValueOracle
from qiskit.optimization.results import GroverOptimizationResults
from qiskit.optimization.util import get_qubo_solutions
from qiskit.aqua.algorithms.amplitude_amplifiers.grover import Grover
from qiskit.visualization import plot_histogram
from qiskit import Aer, execute


class GroverMinimumFinder:

    """Uses Grover Adaptive Search (GAS) to find the minimum of a QUBO function."""

    def __init__(self, num_iterations=3, backend=Aer.get_backend('statevector_simulator'),
                 verbose=False):
        """
        Constructor.
        Args:
            num_iterations (int, optional): The number of iterations the algorithm will search with
                no improvement.
            backend (str, optional): Instance of selected backend.
            verbose (bool, optional): Verbose flag - prints/plots state at each iteration of GAS.
        """
        self._n_iterations = num_iterations
        self._verbose = verbose
        self._backend = backend

    def solve(self, quadratic, linear, constant, num_output_qubits):
        """
        Given the coefficients and constants of a QUBO function, find the minimum output value.
        Args:
            quadratic (np.array): The quadratic coefficients of the QUBO.
            linear (np.array): The linear coefficients of the QUBO.
            constant (int): The constant of the QUBO.
            num_output_qubits (int): The number of qubits used to represent the output values.
        Returns:
            GroverOptimizationResults: A results object containing information about the run,
                including the solution.
        """
        # Variables for tracking the optimum.
        optimum_found = False
        optimum_key = math.inf
        optimum_value = math.inf
        threshold = 0
        n_key = len(linear)
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
        opt_prob_converter = OptimizationProblemToNegativeValueOracle(n_value,
                                                                      verbose=self._verbose,
                                                                      backend=self._backend)

        while not optimum_found:
            m = 1
            improvement_found = False

            # Iterate until we measure a negative.
            loops_with_no_improvement = 0
            while not improvement_found:
                # Determine the number of rotations.
                loops_with_no_improvement += 1
                rotation_count = int(np.ceil(random.uniform(0, m-1)))
                rotations += rotation_count

                # Get state preparation operator A and oracle O for the current threshold.
                a_operator, oracle, func_dict = opt_prob_converter.encode(linear, quadratic,
                                                                          constant - threshold)

                # Apply Grover's Algorithm to find values below the threshold.
                if rotation_count > 0:
                    grover = Grover(oracle, init_state=a_operator, num_iterations=rotation_count)
                    circuit = grover.construct_circuit(
                        measurement=self._backend.name() != "statevector_simulator")
                else:
                    circuit = a_operator._circuit

                # Get the next outcome.
                outcome = self.__measure(circuit, n_key, n_value, self._backend,
                                         verbose=self._verbose)
                k = int(outcome[0:n_key], 2)
                v = outcome[n_key:n_key + n_value]

                # Convert the binary string to integer.
                int_v = self.__bin_to_int(v, n_value) + threshold
                v = self.__twos_complement(int_v, n_value)

                if self._verbose:
                    print("Iterations:", rotation_count)
                    print("Outcome:", outcome)
                    print("Value:", v, "=", int_v)

                # If the value is an improvement, we update the iteration parameters (e.g. oracle).
                if int_v < optimum_value:
                    optimum_key = k
                    optimum_value = int_v
                    if self._verbose:
                        print("Current Optimum Key:", optimum_key)
                        print("Current Optimum:", optimum_value)
                    if v.startswith("1"):
                        improvement_found = True
                        threshold = optimum_value
                else:
                    # If we haven't found a better number after the max number of iterations, we assume the optimal.
                    if loops_with_no_improvement >= self._n_iterations:
                        improvement_found = True
                        optimum_found = True

                    # Using Durr-Hoyer's method, increase m.
                    m = int(np.ceil(min(m * 8/7, 2**(n_key / 2))))
                    if self._verbose:
                        print("No Improvement.")
                        print("M:", m)

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

                if self._verbose:
                    print("Operation Count:", operations)
                    print("\n")

        # Get original key and value pairs.
        func_dict[-1] = constant
        solutions = get_qubo_solutions(func_dict, n_key)

        # If the constant is 0 and we didn't find a negative, the answer is likely 0.
        if optimum_value >= 0 and constant == 0:
            optimum_key = 0

        return GroverOptimizationResults(optimum_key, solutions[optimum_key], operation_count,
                                         rotations, n_key, n_value, func_dict)

    def __measure(self, circuit, n_key, n_value, backend, shots=1024, verbose=False):
        """Get probabilities from the given backend, and picks a random outcome."""
        probs = self.__get_probs(n_key, n_value, circuit, backend, shots)
        freq = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        # Pick a random outcome.
        freq[len(freq)-1] = (freq[len(freq)-1][0], 1 - sum([x[1] for x in freq[0:len(freq)-1]]))
        idx = np.random.choice(len(freq), 1, p=[x[1] for x in freq])[0]
        if verbose:
            print("Frequencies:", freq)
            outcomes = {}
            for label in probs:
                key = str(int(label[:n_key], 2)) + " -> " +\
                      str(self.__bin_to_int(label[n_key:n_key + n_value], n_value))
                outcomes[key] = probs[label]
            plot_histogram(outcomes).show()

        return freq[idx][0]

    @staticmethod
    def __get_probs(n_key, n_value, qc, backend, shots):
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
    def __twos_complement(v, n_bits):
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
    def __bin_to_int(v, num_value_bits):
        """Converts a binary string of n bits using two's complement to an integer."""
        if v.startswith("1"):
            int_v = int(v, 2) - 2 ** num_value_bits
        else:
            int_v = int(v, 2)

        return int_v
