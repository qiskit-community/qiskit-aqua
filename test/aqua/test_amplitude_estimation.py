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
from parameterized import parameterized
from qiskit import QuantumRegister, QuantumCircuit, BasicAer, execute

from test.aqua.common import QiskitAquaTestCase
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.uncertainty_models import LogNormalDistribution, MultivariateNormalDistribution
from qiskit.aqua.components.uncertainty_models import GaussianConditionalIndependenceModel as GCI
from qiskit.aqua.components.uncertainty_problems import EuropeanCallDelta, FixedIncomeExpectedValue
from qiskit.aqua.components.uncertainty_problems import UnivariatePiecewiseLinearObjective as PwlObjective
from qiskit.aqua.components.uncertainty_problems import UnivariateProblem, MultivariateProblem, UncertaintyProblem
from qiskit.aqua.circuits import WeightedSumOperator
from qiskit.aqua.algorithms import AmplitudeEstimation, MaximumLikelihoodAmplitudeEstimation
from qiskit.aqua.algorithms.single_sample.amplitude_estimation.q_factory import QFactory


class BernoulliAFactory(UncertaintyProblem):
    """
    Circuit Factory representing the operator A.
    A is used to initialize the state as well as to construct Q.
    """

    def __init__(self, probability=0.5):
        #
        super().__init__(1)
        self._probability = probability
        self.i_state = 0
        self._theta_p = 2 * np.arcsin(np.sqrt(probability))

    def build(self, qc, q, q_ancillas=None):
        # A is a rotation of angle theta_p around the Y-axis
        qc.ry(self._theta_p, q[self.i_state])

    def value_to_estimation(self, value):
        return value


class BernoulliQFactory(QFactory):
    """
    Circuit Factory representing the operator Q.
    This implementation exploits the fact that powers of Q can be implemented efficiently by just multiplying the angle.
    (amplitude estimation only requires controlled powers of Q, thus, only this method is overridden.)
    """

    def __init__(self, bernoulli_expected_value):
        super().__init__(bernoulli_expected_value, i_objective=0)

    def build(self, qc, q, q_ancillas=None):
        i_state = self.a_factory.i_state
        theta_p = self.a_factory._theta_p
        # Q is a rotation of angle 2*theta_p around the Y-axis
        qc.ry(2 * theta_p, q[i_state])

    def build_power(self, qc, q, power, q_ancillas=None, use_basis_gates=True):
        i_state = self.a_factory.i_state
        theta_p = self.a_factory._theta_p
        qc.ry(2 * power * theta_p, q[i_state])

    def build_controlled_power(self, qc, q, q_control, power, q_ancillas=None, use_basis_gates=True):
        i_state = self.a_factory.i_state
        theta_p = self.a_factory._theta_p
        qc.cry(2 * power * theta_p, q_control, q[i_state])


class TestBernoulli(QiskitAquaTestCase):
    def setUp(self):
        super().setUp()

        self._statevector = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                            circuit_caching=False, seed_simulator=2, seed_transpiler=2)

        def qasm(shots=100):
            return QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'), shots=shots,
                                   circuit_caching=False, seed_simulator=2, seed_transpiler=2)

        self._qasm = qasm

    @parameterized.expand([
        [0.2, AmplitudeEstimation(2), {'estimation': 0.5, 'mle': 0.2}],
        [0.4, AmplitudeEstimation(4), {'estimation': 0.30866, 'mle': 0.4}],
        [0.82, AmplitudeEstimation(5), {'estimation': 0.85355, 'mle': 0.82}],
        [0.49, AmplitudeEstimation(3), {'estimation': 0.5, 'mle': 0.49}],
        [0.2, MaximumLikelihoodAmplitudeEstimation(2), {'estimation': 0.2}],
        [0.4, MaximumLikelihoodAmplitudeEstimation(4), {'estimation': 0.4}],
        [0.82, MaximumLikelihoodAmplitudeEstimation(5), {'estimation': 0.82}],
        [0.49, MaximumLikelihoodAmplitudeEstimation(3), {'estimation': 0.49}]
    ])
    def test_statevector(self, p, ae, expect):
        # construct factories for A and Q
        ae.a_factory = BernoulliAFactory(p)
        ae.q_factory = BernoulliQFactory(ae.a_factory)

        result = ae.run(self._statevector)

        for key, value in expect.items():
            self.assertAlmostEqual(value, result[key], places=3,
                                   msg="estimate `{}` failed".format(key))

    @parameterized.expand([
        [0.2, 100, AmplitudeEstimation(4), {'estimation': 0.14644, 'mle': 0.193888}],
        [0.0, 1000, AmplitudeEstimation(2), {'estimation': 0.0, 'mle': 0.0}],
        [0.8, 10, AmplitudeEstimation(7), {'estimation': 0.79784, 'mle': 0.801612}],
        [0.2, 100, MaximumLikelihoodAmplitudeEstimation(4), {'estimation': 0.199606}],
        [0.4, 1000, MaximumLikelihoodAmplitudeEstimation(6), {'estimation': 0.399488}],
        [0.8, 10, MaximumLikelihoodAmplitudeEstimation(7), {'estimation': 0.800926}]
    ])
    def test_qasm(self, p, shots, ae, expect):
        # construct factories for A and Q
        ae.a_factory = BernoulliAFactory(p)
        ae.q_factory = BernoulliQFactory(ae.a_factory)

        result = ae.run(self._qasm(shots))

        for key, value in expect.items():
            self.assertAlmostEqual(value, result[key], places=3,
                                   msg="estimate `{}` failed".format(key))


class TestEuropeanCallOption(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()

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
        strike_price = 1.896

        # set the approximation scaling for the payoff function
        c_approx = 0.1

        # setup piecewise linear objective fcuntion
        breakpoints = [uncertainty_model.low, strike_price]
        slopes = [0, 1]
        offsets = [0, 0]
        f_min = 0
        f_max = uncertainty_model.high - strike_price
        european_call_objective = PwlObjective(
            uncertainty_model.num_target_qubits,
            uncertainty_model.low,
            uncertainty_model.high,
            breakpoints,
            slopes,
            offsets,
            f_min,
            f_max,
            c_approx
        )

        # construct circuit factory for payoff function
        self.european_call = UnivariateProblem(
            uncertainty_model,
            european_call_objective
        )

        # construct circuit factory for payoff function
        self.european_call_delta = EuropeanCallDelta(
            uncertainty_model,
            strike_price=strike_price,
        )

        self._statevector = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                            circuit_caching=False, seed_simulator=2, seed_transpiler=2)
        self._qasm = QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'), shots=100,
                                     circuit_caching=False, seed_simulator=2, seed_transpiler=2)

    @parameterized.expand([
        ['statevector', AmplitudeEstimation(3), {'estimation': 0.45868536404797905, 'mle': 0.1633160}],
        ['qasm', AmplitudeEstimation(4), {'estimation': 0.45868536404797905, 'mle': 0.23479973342434832}],
        ['statevector', MaximumLikelihoodAmplitudeEstimation(5), {'estimation': 0.16330976193204114}],
        ['qasm', MaximumLikelihoodAmplitudeEstimation(3), {'estimation': 0.1027255930905642}],
    ])
    def test_expected_value(self, simulator, ae, expect):

        # set A factory for amplitude estimation
        ae.a_factory = self.european_call

        # run simulation
        result = ae.run(self._qasm if simulator == 'qasm' else self._statevector)

        # compare to precomputed solution
        for key, value in expect.items():
            self.assertAlmostEqual(result[key], value, places=4,
                                   msg="estimate `{}` failed".format(key))

    @parameterized.expand([
        ['statevector', AmplitudeEstimation(3), {'estimation': 0.8535534, 'mle': 0.8097974047170567}],
        ['qasm', AmplitudeEstimation(4), {'estimation': 0.8535534, 'mle': 0.8143597808556013}],
        ['statevector', MaximumLikelihoodAmplitudeEstimation(5), {'estimation': 0.8097582003326866}],
        ['qasm', MaximumLikelihoodAmplitudeEstimation(6), {'estimation': 0.8096123776923358}],
    ])
    def test_delta(self, simulator, ae, expect):
        # set A factory for amplitude estimation
        ae.a_factory = self.european_call_delta

        # run simulation
        result = ae.run(self._qasm if simulator == 'qasm' else self._statevector)

        # compare to precomputed solution
        for key, value in expect.items():
            self.assertAlmostEqual(result[key], value, places=4,
                                   msg="estimate `{}` failed".format(key))


class TestFixedIncomeAssets(QiskitAquaTestCase):
    def setUp(self):
        super().setUp()

        self._statevector = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                            circuit_caching=False, seed_simulator=2, seed_transpiler=2)
        self._qasm = QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'), shots=100,
                                     circuit_caching=False, seed_simulator=2, seed_transpiler=2)

    @parameterized.expand([
        ['statevector', AmplitudeEstimation(5), {'estimation': 2.4600, 'mle': 2.3402315559106843}],
        ['qasm', AmplitudeEstimation(5), {'estimation': 2.4600, 'mle': 2.3632087675061726}],
        ['statevector', MaximumLikelihoodAmplitudeEstimation(5), {'estimation': 2.340361798381051}],
        ['qasm', MaximumLikelihoodAmplitudeEstimation(5), {'estimation': 2.317921060790118}]
    ])
    def test_expected_value(self, simulator, ae, expect):
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
        ae.a_factory = fixed_income

        # run simulation
        result = ae.run(self._qasm if simulator == 'qasm' else self._statevector)

        # compare to precomputed solution
        for key, value in expect.items():
            self.assertAlmostEqual(result[key], value, places=4,
                                   msg="estimate `{}` failed".format(key))


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
        # z_values = np.linspace(-z_max, z_max, 2 ** n_z)
        p_zeros = [0.15, 0.25]
        rhos = [0.1, 0.05]
        lgd = [1, 2]
        K = len(p_zeros)
        # alpha = 0.05

        # set var value
        var = 2
        var_prob = 0.961940

        # determine number of qubits required to represent total loss
        # n_s = WeightedSumOperator.get_required_sum_qubits(lgd)

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
