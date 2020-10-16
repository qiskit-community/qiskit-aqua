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
from qiskit.circuit.library import IntegerComparator, LogNormalDistribution
from qiskit.quantum_info import Operator

from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import IterativeAmplitudeEstimation
from qiskit.finance.applications import EuropeanCallDelta


class TestEuropeanCallDelta(QiskitFinanceTestCase):
    """Tests European Call Expected Value uncertainty problem """

    def test_circuit(self):
        """Test the expected circuit.

        If it equals the correct ``IntegerComparator`` we know the circuit is correct.
        """
        num_qubits = 3
        strike_price = 0.5
        bounds = (0, 2)
        ecd = EuropeanCallDelta(num_qubits, strike_price, bounds)

        # map strike_price to a basis state
        x = (strike_price - bounds[0]) / (bounds[1] - bounds[0]) * (2 ** num_qubits - 1)
        comparator = IntegerComparator(num_qubits, x)

        self.assertTrue(Operator(ecd).equiv(comparator))

    def test_application(self):
        """Test an end-to-end application."""
        num_qubits = 3

        # parameters for considered random distribution
        s_p = 2.0  # initial spot price
        vol = 0.4  # volatility of 40%
        r = 0.05  # annual interest rate of 4%
        t_m = 40 / 365  # 40 days to maturity

        # resulting parameters for log-normal distribution
        mu = ((r - 0.5 * vol ** 2) * t_m + np.log(s_p))
        sigma = vol * np.sqrt(t_m)
        mean = np.exp(mu + sigma ** 2 / 2)
        variance = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
        stddev = np.sqrt(variance)

        # lowest and highest value considered for the spot price;
        # in between, an equidistant discretization is considered.
        low = np.maximum(0, mean - 3 * stddev)
        high = mean + 3 * stddev
        bounds = (low, high)

        # construct circuit factory for uncertainty model
        uncertainty_model = LogNormalDistribution(num_qubits,
                                                  mu=mu, sigma=sigma ** 2, bounds=bounds)

        # set the strike price (should be within the low and the high value of the uncertainty)
        strike_price = 1.896

        # create amplitude function
        european_call_delta = EuropeanCallDelta(num_qubits, strike_price, bounds)

        # create state preparation
        state_preparation = european_call_delta.compose(uncertainty_model, front=True)

        # run amplitude estimation
        iae = IterativeAmplitudeEstimation(0.01, 0.05, state_preparation=state_preparation,
                                           objective_qubits=[num_qubits])

        backend = QuantumInstance(Aer.get_backend('qasm_simulator'),
                                  seed_simulator=125, seed_transpiler=80)
        result = iae.run(backend)
        self.assertAlmostEqual(result.estimation, 0.8079816552117238)


if __name__ == '__main__':
    unittest.main()
