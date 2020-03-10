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

""" Test Optimization Problem to Negative Value Oracle """

from test.optimization import QiskitOptimizationTestCase
import numpy as np
from qiskit.optimization.converters import OptimizationProblemToNegativeValueOracle
from qiskit.optimization.util import get_qubo_solutions
from qiskit import QuantumCircuit, Aer, execute


class TestOptimizationProblemToNegativeValueOracle(QiskitOptimizationTestCase):
    """OPtNVO Tests"""

    def _validate_function(self, func_dict, linear, quadratic, constant):
        for key in func_dict:
            if isinstance(key, int) and key >= 0:
                self.assertEqual(-1 * linear[key], func_dict[key])
            elif isinstance(key, tuple):
                self.assertEqual(quadratic[key[0]][key[1]], func_dict[key])
            else:
                self.assertEqual(constant, func_dict[key])

    def _validate_operator(self, func_dict, n_key, n_value, operator):
        # Get expected results.
        solutions = get_qubo_solutions(func_dict, n_key, print_solutions=False)

        # Run the state preparation operator A and observe results.
        circuit = operator._circuit
        qc = QuantumCircuit() + circuit
        hist = self._measure(qc, n_key, n_value)

        # Validate operator A.
        for label in hist:
            key = int(label[:n_key], 2)
            value = self._bin_to_int(label[n_key:n_key + n_value], n_value)
            self.assertEqual(int(solutions[key]), value)

    @staticmethod
    def _measure(qc, n_key, n_value):
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend=backend, shots=1)
        result = job.result()
        state = np.round(result.get_statevector(qc), 5)
        keys = [bin(i)[2::].rjust(int(np.log2(len(state))), '0')[::-1]
                for i in range(0, len(state))]
        probs = [np.round(abs(a)*abs(a), 5) for a in state]
        f_hist = dict(zip(keys, probs))
        hist = {}
        for key in f_hist:
            new_key = key[:n_key] + key[n_key:n_key+n_value][::-1] + key[n_key+n_value:]
            hist[new_key] = f_hist[key]
        hist = dict(filter(lambda p: p[1] > 0, hist.items()))
        return hist

    @staticmethod
    def _bin_to_int(v, num_value_bits):
        if v.startswith("1"):
            int_v = int(v, 2) - 2 ** num_value_bits
        else:
            int_v = int(v, 2)

        return int_v

    def test_optnvo_2_key(self):
        """ Test with 2 linear coefficients, no quadratic or constant """
        # Circuit parameters.
        num_value = 4

        # Input.
        linear = np.array([1, -2])
        quadratic = np.array([[1, 0],
                              [0, 1]])
        constant = 0

        # Convert to dictionary format with operator/oracle.
        converter = OptimizationProblemToNegativeValueOracle(num_value)
        a_operator, _, func_dict = converter.encode(linear, quadratic, constant)

        self._validate_function(func_dict, linear, quadratic, constant)
        self._validate_operator(func_dict, len(linear), num_value, a_operator)

    def test_optnvo_2_key_w_constant(self):
        """ Test with 2 linear coefficients, no quadratic, simple constant """
        # Circuit parameters.
        num_value = 4

        # Input.
        linear = np.array([1, -2])
        quadratic = np.array([[1, 0],
                              [0, 1]])
        constant = 1

        # Convert to dictionary format with operator/oracle.
        converter = OptimizationProblemToNegativeValueOracle(num_value)
        a_operator, _, func_dict = converter.encode(linear, quadratic, constant)

        self._validate_function(func_dict, linear, quadratic, constant)
        self._validate_operator(func_dict, len(linear), num_value, a_operator)

    def test_optnvo_4_key_all_negative(self):
        """ Test with all negative values """
        # Circuit parameters.
        num_value = 5

        # Input.
        linear = np.array([1, 1, 1, 1])
        quadratic = np.array([[-1, -1, -1, -1],
                              [-1, -1, -1, -1],
                              [-1, -1, -1, -1],
                              [-1, -1, -1, -1]])
        constant = -1

        # Convert to dictionary format with operator/oracle.
        converter = OptimizationProblemToNegativeValueOracle(num_value)
        a_operator, _, func_dict = converter.encode(linear, quadratic, constant)

        self._validate_function(func_dict, linear, quadratic, constant)
        self._validate_operator(func_dict, len(linear), num_value, a_operator)

    def test_optnvo_6_key(self):
        """ Test with 6 linear coefficients, negative quadratics, no constant """
        # Circuit parameters.
        num_value = 4

        # Input.
        linear = np.array([1, -2, -1, 0, 1, 2])
        quadratic = np.array([[1, 0, 0, -1, 0, 0],
                              [0, 1, 0, 0, 0, -2],
                              [0, 0, 1, 0, 0, 0],
                              [-1, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, -2, 0, 0, 0, 1]])
        constant = 0

        # Convert to dictionary format with operator/oracle.
        converter = OptimizationProblemToNegativeValueOracle(num_value)
        a_operator, _, func_dict = converter.encode(linear, quadratic, constant)

        self._validate_function(func_dict, linear, quadratic, constant)
        self._validate_operator(func_dict, len(linear), num_value, a_operator)
