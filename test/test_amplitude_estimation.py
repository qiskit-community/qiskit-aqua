# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest

import numpy as np

from test.common import QiskitAquaTestCase

from parameterized import parameterized

from qiskit import QuantumRegister, QuantumCircuit, BasicAer, execute

from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.uncertainty_models import LogNormalDistribution, MultivariateNormalDistribution
from qiskit.aqua.components.uncertainty_models import GaussianConditionalIndependenceModel as GCI
from qiskit.aqua.components.uncertainty_problems import EuropeanCallExpectedValue, EuropeanCallDelta, FixedIncomeExpectedValue
from qiskit.aqua.components.uncertainty_problems import UnivariatePiecewiseLinearObjective as PwlObjective
from qiskit.aqua.components.uncertainty_problems import MultivariateProblem
from qiskit.aqua.circuits import WeightedSumOperator
from qiskit.aqua.algorithms import AmplitudeEstimation


class TestEuropeanCallOption(QiskitAquaTestCase):

    @parameterized.expand([
        'qasm_simulator',
        'statevector_simulator'
    ])
    def test_expected_value(self, simulator):

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

        # run simulation
        quantum_instance = QuantumInstance(BasicAer.get_backend(simulator))
        result = ae.run(quantum_instance=quantum_instance)

        # compare to precomputed solution
        self.assertEqual(0.0, np.round(result['estimation'] - 0.045705353233, decimals=4))

    @parameterized.expand([
        'qasm_simulator',
        'statevector_simulator'
    ])
    def test_delta(self, simulator):

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

        # run simulation
        quantum_instance = QuantumInstance(BasicAer.get_backend(simulator))
        result = ae.run(quantum_instance=quantum_instance)

        # compare to precomputed solution
        self.assertEqual(0.0, np.round(result['estimation'] - 0.5000, decimals=4))


class TestFixedIncomeAssets(QiskitAquaTestCase):

    @parameterized.expand([
        'qasm_simulator',
        'statevector_simulator'
    ])
    def test_expected_value(self, simulator):

        # can be used in case a principal component analysis has been done to derive the uncertainty model, ignored in this example.
        A = np.eye(2)
        b = np.zeros(2)

        # specify the number of qubits that are used to represent the different dimenions of the uncertainty model
        num_qubits = [2, 2]

        # specify the lower and upper bounds for the different dimension
        low = [0, 0]
        high = [0.12, 0.24]
        mu = [0.12, 0.24]
        sigma = 0.01 * np.eye(2)

        # construct corresponding distribution
        u = MultivariateNormalDistribution(num_qubits, low, high, mu, sigma)

        # specify cash flow
        cf = [1.0, 2.0]

        # specify approximation factor
        c_approx = 0.125

        # get fixed income circuit appfactory
        fixed_income = FixedIncomeExpectedValue(u, A, b, cf, c_approx)

        # set number of evaluation qubits (samples)
        m = 5

        # construct amplitude estimation
        ae = AmplitudeEstimation(m, fixed_income)

        # run simulation
        quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'))
        result = ae.run(quantum_instance=quantum_instance)

        # compare to precomputed solution
        self.assertEqual(0.0, np.round(result['estimation'] - 2.4600, decimals=4))


class TestCreditRiskAnalysis(QiskitAquaTestCase):

    @parameterized.expand([
        'statevector_simulator'
    ])
    def test_conditional_value_at_risk(self, simulator):

        # define backend to be used
        backend = BasicAer.get_backend(simulator)

        # set problem parameters
        n_z = 2
        z_max = 2
        z_values = np.linspace(-z_max, z_max, 2 ** n_z)
        p_zeros = [0.15, 0.25]
        rhos = [0.1, 0.05]
        lgd = [1, 2]
        K = len(p_zeros)
        alpha = 0.05

        # set var value
        var = 2
        var_prob = 0.961940

        # determine number of qubits required to represent total loss
        n_s = WeightedSumOperator.get_required_sum_qubits(lgd)

        # create circuit factory (add Z qubits with weight/loss 0)
        agg = WeightedSumOperator(n_z + K, [0] * n_z + lgd)

        # define linear objective
        breakpoints = [0, var]
        slopes = [0, 1]
        offsets = [0, 0]  # subtract VaR and add it later to the estimate
        f_min = 0
        f_max = 3 - var
        c_approx = 0.25

        # construct circuit factory for uncertainty model (Gaussian Conditional Independence model)
        u = GCI(n_z, z_max, p_zeros, rhos)

        cvar_objective = PwlObjective(
            agg.num_sum_qubits,
            0,
            2 ** agg.num_sum_qubits - 1,  # max value that can be reached by the qubit register (will not always be reached)
            breakpoints,
            slopes,
            offsets,
            f_min,
            f_max,
            c_approx
        )

        multivariate_cvar = MultivariateProblem(u, agg, cvar_objective)

        num_qubits = multivariate_cvar.num_target_qubits
        num_ancillas = multivariate_cvar.required_ancillas()

        q = QuantumRegister(num_qubits, name='q')
        q_a = QuantumRegister(num_ancillas, name='q_a')
        qc = QuantumCircuit(q, q_a)

        multivariate_cvar.build(qc, q, q_a)

        job = execute(qc, backend=backend)

        # evaluate resulting statevector
        value = 0
        for i, a in enumerate(job.result().get_statevector()):
            b = ('{0:0%sb}' % multivariate_cvar.num_target_qubits).format(i)[-multivariate_cvar.num_target_qubits:]
            am = np.round(np.real(a), decimals=4)
            if np.abs(am) > 1e-6 and b[0] == '1':
                value += am ** 2

        # normalize and add VaR to estimate
        value = multivariate_cvar.value_to_estimation(value)
        normalized_value = value / (1.0 - var_prob) + var

        # compare to precomputed solution
        self.assertEqual(0.0, np.round(normalized_value - 3.3796, decimals=4))

if __name__ == '__main__':
    unittest.main()
