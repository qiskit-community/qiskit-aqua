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

""" Test European Call Expected Value uncertainty problem """

import unittest
from test.finance import QiskitFinanceTestCase

import numpy as np

from qiskit import QuantumCircuit
from qiskit.finance.applications import FixedIncomeExpectedValue
from qiskit.quantum_info import Operator


class TestFixedIncomeExpectedValue(QiskitFinanceTestCase):
    """Tests European Call Expected Value uncertainty problem """

    # TODO add test that checks the function on amplitudes
    # def assertFunctionIsCorrect(self, function_circuit, reference):
    #     """Assert that ``function_circuit`` implements the reference function ``reference``."""
    #     num_state_qubits = function_circuit.num_qubits - 1

    #     circuit = QuantumCircuit(function_circuit.num_qubits)
    #     circuit.h(list(range(num_state_qubits)))
    #     circuit.append(function_circuit.to_instruction(), list(range(circuit.num_qubits)))

    #     backend = BasicAer.get_backend('statevector_simulator')
    #     statevector = execute(circuit, backend).result().get_statevector()

    #     expected = []
    #     for i, _ in enumerate(statevector):
    #         state = bin(i)[2:].zfill(num_state_qubits + 1)
    #         x, last_qubit = int(state[1:], 2), state[0]
    #         if last_qubit == '0':
    #             expected_amplitude = np.cos(reference(x)) / np.sqrt(2**num_state_qubits)
    #         else:
    #             expected_amplitude = np.sin(reference(x)) / np.sqrt(2**num_state_qubits)

    #         expected += [expected_amplitude]

    #     print(expected)
    #     print(statevector)

    # np.testing.assert_almost_equal(unrolled_probabilities, unrolled_expectations)

    def test_circuit(self):
        """Test the expected circuit."""
        num_qubits = [2, 2]
        pca = np.eye(2)
        initial_interests = np.zeros(2)
        cash_flow = np.array([1, 2])
        rescaling_factor = 0.125
        bounds = [(0, 0.12), (0, 0.24)]

        circuit = FixedIncomeExpectedValue(num_qubits, pca, initial_interests, cash_flow,
                                           rescaling_factor, bounds)

        expected = QuantumCircuit(5)
        expected.cry(-np.pi / 216, 0, 4)
        expected.cry(-np.pi / 108, 1, 4)
        expected.cry(-np.pi / 27, 2, 4)
        expected.cry(-0.23271, 3, 4)
        expected.ry(9 * np.pi / 16, 4)

        self.assertTrue(Operator(circuit).equiv(expected))


if __name__ == '__main__':
    unittest.main()
