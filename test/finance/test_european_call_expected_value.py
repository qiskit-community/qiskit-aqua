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

from qiskit import Aer
from qiskit.circuit.library import TwoLocal, NormalDistribution
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import IterativeAmplitudeEstimation
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit.finance.applications import EuropeanCallExpectedValue
from qiskit.quantum_info import Operator


class TestEuropeanCallExpectedValue(QiskitFinanceTestCase):
    """Tests European Call Expected Value uncertainty problem """

    def setUp(self):
        super().setUp()
        self.seed = 457
        aqua_globals.random_seed = self.seed

    def test_ecev_circuit(self):
        """Test the expected circuit.

        If it equals the correct ``LinearAmplitudeFunction`` we know the circuit is correct.
        """
        num_qubits = 3
        rescaling_factor = 0.1
        strike_price = 0.5
        bounds = (0, 2)
        ecev = EuropeanCallExpectedValue(num_qubits, strike_price, rescaling_factor, bounds)

        breakpoints = [0, strike_price]
        slopes = [0, 1]
        offsets = [0, 0]
        image = (0, 2 - strike_price)
        domain = (0, 2)
        linear_function = LinearAmplitudeFunction(
            num_qubits,
            slopes,
            offsets,
            domain=domain,
            image=image,
            breakpoints=breakpoints,
            rescaling_factor=rescaling_factor)

        self.assertTrue(Operator(ecev).equiv(linear_function))

    def test_application(self):
        """Test an end-to-end application."""
        bounds = np.array([0., 7.])
        num_qubits = 3

        # the distribution circuit is a normal distribution plus a QGAN-trained ansatz circuit
        dist = NormalDistribution(num_qubits, mu=1, sigma=1, bounds=bounds)

        ansatz = TwoLocal(num_qubits, 'ry', 'cz', reps=1, entanglement='circular')
        trained_params = [0.29399714, 0.38853322, 0.9557694, 0.07245791, 6.02626428, 0.13537225]
        ansatz.assign_parameters(trained_params, inplace=True)

        dist.compose(ansatz, inplace=True)

        # create the European call expected value
        strike_price = 2
        rescaling_factor = 0.25
        european_call = EuropeanCallExpectedValue(num_qubits, strike_price, rescaling_factor,
                                                  bounds)

        # create the state preparation circuit
        state_preparation = european_call.compose(dist, front=True)

        iae = IterativeAmplitudeEstimation(0.01, 0.05, state_preparation=state_preparation,
                                           objective_qubits=[num_qubits],
                                           post_processing=european_call.post_processing)

        backend = QuantumInstance(Aer.get_backend('qasm_simulator'),
                                  seed_simulator=125, seed_transpiler=80)
        result = iae.run(backend)
        self.assertAlmostEqual(result.estimation, 1.0127253837345427)


if __name__ == '__main__':
    unittest.main()
