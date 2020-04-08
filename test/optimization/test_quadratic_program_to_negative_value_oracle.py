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

"""Test Optimization Problem to Negative Value Oracle."""

import unittest
from test.optimization import QiskitOptimizationTestCase
import numpy as np
from qiskit.optimization.converters import QuadraticProgramToNegativeValueOracle
from qiskit.optimization.util import get_qubo_solutions
from qiskit import QuantumCircuit, Aer, execute
from qiskit.optimization.problems import QuadraticProgram


class TestQuadraticProgramToNegativeValueOracle(QiskitOptimizationTestCase):
    """OPtNVO Tests"""

    def _validate_function(self, func_dict, problem):
        linear = problem.objective.get_linear_dict()
        quadratic = problem.objective.get_quadratic_dict()
        for key in func_dict:
            if isinstance(key, int) and key >= 0:
                self.assertEqual(linear[key], func_dict[key])
            elif isinstance(key, tuple):
                self.assertEqual(quadratic[key[0], key[1]], func_dict[key])
            else:
                self.assertEqual(problem.objective.get_offset(), func_dict[key])

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
        if v.startswith('1'):
            int_v = int(v, 2) - 2 ** num_value_bits
        else:
            int_v = int(v, 2)

        return int_v

    def test_optnvo_3_linear_2_quadratic_no_constant(self):
        """Test with 3 linear coefficients, 2 quadratic, and no constant."""
        # Circuit parameters.
        num_value = 4

        # Input.
        problem = QuadraticProgram()
        problem.variables.add(names=['x0', 'x1', 'x2'], types='BBB')
        linear = [('x0', -1), ('x1', 2), ('x2', -3)]
        problem.objective.set_linear(linear)
        problem.objective.set_quadratic_coefficients('x0', 'x2', -2)
        problem.objective.set_quadratic_coefficients('x1', 'x2', -1)

        # Convert to dictionary format with operator/oracle.
        converter = QuadraticProgramToNegativeValueOracle(num_value)
        a_operator, _, func_dict = converter.encode(problem)

        self._validate_function(func_dict, problem)
        self._validate_operator(func_dict, len(linear), num_value, a_operator)

    def test_optnvo_4_key_all_negative(self):
        """Test with all negative values."""
        # Circuit parameters.
        num_value = 5

        # Input.
        problem = QuadraticProgram()
        problem.variables.add(names=['x0', 'x1', 'x2'], types='BBB')
        linear = [('x0', -1), ('x1', -2), ('x2', -1)]
        problem.objective.set_linear(linear)
        problem.objective.set_quadratic_coefficients('x0', 'x1', -1)
        problem.objective.set_quadratic_coefficients('x0', 'x2', -2)
        problem.objective.set_quadratic_coefficients('x1', 'x2', -1)
        problem.objective.set_offset(-1)

        # Convert to dictionary format with operator/oracle.
        converter = QuadraticProgramToNegativeValueOracle(num_value)
        a_operator, _, func_dict = converter.encode(problem)

        self._validate_function(func_dict, problem)
        self._validate_operator(func_dict, len(linear), num_value, a_operator)

    def test_optnvo_6_key(self):
        """Test with 6 linear coefficients, negative quadratics, no constant."""
        # Circuit parameters.
        num_value = 4

        # Input.
        problem = QuadraticProgram()
        problem.variables.add(names=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], types='BBBBBB')
        linear = [('x0', -1), ('x1', -2), ('x2', -1), ('x3', 0), ('x4', 1), ('x5', 2)]
        problem.objective.set_linear(linear)
        problem.objective.set_quadratic_coefficients('x0', 'x3', -1)
        problem.objective.set_quadratic_coefficients('x1', 'x5', -2)

        # Convert to dictionary format with operator/oracle.
        converter = QuadraticProgramToNegativeValueOracle(num_value)
        a_operator, _, func_dict = converter.encode(problem)

        self._validate_function(func_dict, problem)
        self._validate_operator(func_dict, len(linear), num_value, a_operator)


if __name__ == '__main__':
    unittest.main()
