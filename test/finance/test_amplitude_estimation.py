# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Amplitude Estimation """

import unittest
import warnings
from test.finance import QiskitFinanceTestCase
import numpy as np
from ddt import ddt, idata, unpack
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.uncertainty_models import (LogNormalDistribution,
                                                       MultivariateNormalDistribution)
from qiskit.finance.components.uncertainty_problems import (EuropeanCallDelta,
                                                            FixedIncomeExpectedValue)
from qiskit.aqua.components.uncertainty_problems import \
    UnivariatePiecewiseLinearObjective as PwlObjective
from qiskit.aqua.components.uncertainty_problems import UnivariateProblem
from qiskit.aqua.algorithms import AmplitudeEstimation, MaximumLikelihoodAmplitudeEstimation


@ddt
class TestEuropeanCallOption(QiskitFinanceTestCase):
    """ Test European Call Option """

    def setUp(self):
        super().setUp()

        # number of qubits to represent the uncertainty
        num_uncertainty_qubits = 3

        # parameters for considered random distribution
        s_p = 2.0  # initial spot price
        vol = 0.4  # volatility of 40%
        r = 0.05  # annual interest rate of 4%
        t_m = 40 / 365  # 40 days to maturity

        # resulting parameters for log-normal distribution
        m_u = ((r - 0.5 * vol ** 2) * t_m + np.log(s_p))
        sigma = vol * np.sqrt(t_m)
        mean = np.exp(m_u + sigma ** 2 / 2)
        variance = (np.exp(sigma ** 2) - 1) * np.exp(2 * m_u + sigma ** 2)
        stddev = np.sqrt(variance)

        # lowest and highest value considered for the spot price;
        # in between, an equidistant discretization is considered.
        low = np.maximum(0, mean - 3 * stddev)
        high = mean + 3 * stddev

        # construct circuit factory for uncertainty model
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        uncertainty_model = LogNormalDistribution(num_uncertainty_qubits,
                                                  mu=m_u, sigma=sigma, low=low, high=high)

        # set the strike price (should be within the low and the high value of the uncertainty)
        strike_price = 1.896

        # set the approximation scaling for the payoff function
        c_approx = 0.1

        # setup piecewise linear objective function
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
        warnings.filterwarnings('always', category=DeprecationWarning)

        self._statevector = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                            seed_simulator=2,
                                            seed_transpiler=2)
        self._qasm = QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'), shots=100,
                                     seed_simulator=2, seed_transpiler=2)

    @idata([
        ['statevector', AmplitudeEstimation(3),
         {'estimation': 0.45868536404797905, 'mle': 0.1633160}],
        ['qasm', AmplitudeEstimation(4),
         {'estimation': 0.45868536404797905, 'mle': 0.23479973342434832}],
        ['statevector', MaximumLikelihoodAmplitudeEstimation(5),
         {'estimation': 0.16330976193204114}],
        ['qasm', MaximumLikelihoodAmplitudeEstimation(3),
         {'estimation': 0.09784548904622023}],
    ])
    @unpack
    def test_expected_value(self, simulator, a_e, expect):
        """ expected value test """
        # set A factory for amplitude estimation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            a_e.a_factory = self.european_call

        # run simulation
        result = a_e.run(self._qasm if simulator == 'qasm' else self._statevector)

        # compare to precomputed solution
        for key, value in expect.items():
            self.assertAlmostEqual(getattr(result, key), value, places=4,
                                   msg="estimate `{}` failed".format(key))

    @idata([
        ['statevector', AmplitudeEstimation(3),
         {'estimation': 0.8535534, 'mle': 0.8097974047170567}],
        ['qasm', AmplitudeEstimation(4),
         {'estimation': 0.8535534, 'mle': 0.8143597808556013}],
        ['statevector', MaximumLikelihoodAmplitudeEstimation(5),
         {'estimation': 0.8097582003326866}],
        ['qasm', MaximumLikelihoodAmplitudeEstimation(6),
         {'estimation': 0.8096123776923358}],
    ])
    @unpack
    def test_delta(self, simulator, a_e, expect):
        """ delta test """
        # set A factory for amplitude estimation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            a_e.a_factory = self.european_call_delta

        # run simulation
        result = a_e.run(self._qasm if simulator == 'qasm' else self._statevector)

        # compare to precomputed solution
        for key, value in expect.items():
            self.assertAlmostEqual(getattr(result, key), value, places=4,
                                   msg="estimate `{}` failed".format(key))


@ddt
class TestFixedIncomeAssets(QiskitFinanceTestCase):
    """ Test Fixed Income Assets """

    def setUp(self):
        super().setUp()
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        self._statevector = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                            seed_simulator=2,
                                            seed_transpiler=2)
        self._qasm = QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'),
                                     shots=100,
                                     seed_simulator=2,
                                     seed_transpiler=2)

    def tearDown(self):
        super().tearDown()
        warnings.filterwarnings('always', category=DeprecationWarning)

    @idata([
        ['statevector', AmplitudeEstimation(5),
         {'estimation': 2.4600, 'mle': 2.3402315559106843}],
        ['qasm', AmplitudeEstimation(5),
         {'estimation': 2.4600, 'mle': 2.3632087675061726}],
        ['statevector', MaximumLikelihoodAmplitudeEstimation(5),
         {'estimation': 2.340228883624973}],
        ['qasm', MaximumLikelihoodAmplitudeEstimation(5),
         {'estimation': 2.3174630932734077}]
    ])
    @unpack
    def test_expected_value(self, simulator, a_e, expect):
        """ expected value test """
        # can be used in case a principal component analysis
        # has been done to derive the uncertainty model, ignored in this example.
        a_n = np.eye(2)
        b = np.zeros(2)

        # specify the number of qubits that are used to represent
        # the different dimensions of the uncertainty model
        num_qubits = [2, 2]

        # specify the lower and upper bounds for the different dimension
        low = [0, 0]
        high = [0.12, 0.24]
        m_u = [0.12, 0.24]
        sigma = 0.01 * np.eye(2)

        # construct corresponding distribution
        mund = MultivariateNormalDistribution(num_qubits, low, high, m_u, sigma)

        # specify cash flow
        c_f = [1.0, 2.0]

        # specify approximation factor
        c_approx = 0.125

        # get fixed income circuit appfactory
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            fixed_income = FixedIncomeExpectedValue(mund, a_n, b, c_f, c_approx)
            a_e.a_factory = fixed_income

        # run simulation
        result = a_e.run(self._qasm if simulator == 'qasm' else self._statevector)

        # compare to precomputed solution
        for key, value in expect.items():
            self.assertAlmostEqual(getattr(result, key), value, places=4,
                                   msg="estimate `{}` failed".format(key))


if __name__ == '__main__':
    unittest.main()
