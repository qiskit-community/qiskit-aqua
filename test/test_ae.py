# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import unittest
from test.common import QiskitAquaTestCase

import numpy as np
from qiskit_aqua.algorithms.single_sample import AmplitudeEstimation
from qiskit_aqua.components.uncertainty_problems import EuropeanCallExpectedValue, EuropeanCallDelta, FixedIncomeExpectedValue
from qiskit_aqua.components.random_distributions import LogNormalDistribution, MultivariateNormalDistribution
from qiskit import BasicAer


class TestEuropeanCallOption(QiskitAquaTestCase):

    def test_expected_value(self):

        # number of qubits to represent the uncertainty
        num_uncertainty_qubits = 3

        # parameters for considered random distribution
        S = 2.0  # initial spot price
        vol = 0.4  # volatility of 40%
        r = 0.05  # annual interest rate of 4%
        T = 40 / 365  # 40 days to maturity

        # resulting parameters for log-normal distribution
        mu = ((r - 0.5 * vol ** 2) * T + np.log(S))
        sigma = vol * np.sqrt(T)
        mean = np.exp(mu + sigma ** 2 / 2)
        variance = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
        stddev = np.sqrt(variance)

        # lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
        low = np.maximum(0, mean - 3 * stddev)
        high = mean + 3 * stddev

        # construct circuit factory for uncertainty model
        uncertainty_model = LogNormalDistribution(num_uncertainty_qubits, mu=mu, sigma=sigma, low=low, high=high)

        # set the strike price (should be within the low and the high value of the uncertainty)
        strike_price = 2

        # set the approximation scaling for the payoff function
        c_approx = 0.5

        # construct circuit factory for payoff function
        european_call = EuropeanCallExpectedValue(
            uncertainty_model,
            strike_price=strike_price,
            c_approx=c_approx
        )

        # set number of evaluation qubits (samples)
        m = 3

        # construct amplitude estimation
        ae = AmplitudeEstimation(m, european_call)

        # result = ae.run(quantum_instance=Aer.get_backend('qasm_simulator_py'), shots=100)
        result = ae.run(quantum_instance=BasicAer.get_backend('statevector_simulator'))

        self.assertEqual(0.0, np.round(result['estimation'] - 0.045705353233, decimals=4))

    def test_delta(self):

        # number of qubits to represent the uncertainty
        num_uncertainty_qubits = 3

        # parameters for considered random distribution
        S = 2.0  # initial spot price
        vol = 0.4  # volatility of 40%
        r = 0.05  # anual interest rate of 4%
        T = 40 / 365  # 40 days to maturity

        # resulting parameters for log-normal distribution
        mu = ((r - 0.5 * vol ** 2) * T + np.log(S))
        sigma = vol * np.sqrt(T)
        mean = np.exp(mu + sigma ** 2 / 2)
        variance = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
        stddev = np.sqrt(variance)

        # lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
        low = np.maximum(0, mean - 3 * stddev)
        high = mean + 3 * stddev

        # construct circuit factory for uncertainty model
        uncertainty_model = LogNormalDistribution(num_uncertainty_qubits, mu=mu, sigma=sigma, low=low, high=high)

        # set the strike price (should be within the low and the high value of the uncertainty)
        strike_price = 2

        # construct circuit factory for payoff function
        european_call_delta = EuropeanCallDelta(
            uncertainty_model,
            strike_price=strike_price,
        )

        # set number of evaluation qubits (samples)
        m = 3

        # construct amplitude estimation
        ae = AmplitudeEstimation(m, european_call_delta)

        # result = ae.run(quantum_instance=Aer.get_backend('qasm_simulator_py'), shots=100)
        result = ae.run(quantum_instance=BasicAer.get_backend('statevector_simulator'))

        self.assertEqual(0.0, np.round(result['estimation'] - 0.5000, decimals=4))


if __name__ == '__main__':
    unittest.main()
