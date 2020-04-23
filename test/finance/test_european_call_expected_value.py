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

""" Test European Call Expected Value uncertainty problem """

import unittest
from test.finance import QiskitFinanceTestCase
import warnings
from ddt import ddt, data

import numpy as np

from qiskit import BasicAer
from qiskit.circuit import ParameterVector
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import AmplitudeEstimation
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.components.uncertainty_models import (UnivariateVariationalDistribution,
                                                       NormalDistribution)
from qiskit.aqua.components.variational_forms import RY
from qiskit.finance.components.uncertainty_problems import EuropeanCallExpectedValue


@ddt
class TestEuropeanCallExpectedValue(QiskitFinanceTestCase):
    """Tests European Call Expected Value uncertainty problem """

    def setUp(self):
        super().setUp()
        self.seed = 457
        aqua_globals.random_seed = self.seed

    def tearDown(self):
        super().tearDown()
        warnings.filterwarnings(action="always", category=DeprecationWarning)

    @data(False, True)
    def test_ecev(self, use_circuits):
        """ European Call Expected Value test """
        if not use_circuits:
            # ignore deprecation warnings from the deprecation of VariationalForm as input for
            # the univariate variational distribution
            warnings.filterwarnings("ignore", category=DeprecationWarning)

        bounds = np.array([0., 7.])
        num_qubits = [3]
        entangler_map = []
        for i in range(sum(num_qubits)):
            entangler_map.append([i, int(np.mod(i + 1, sum(num_qubits)))])

        g_params = [0.29399714, 0.38853322, 0.9557694, 0.07245791, 6.02626428, 0.13537225]
        # Set an initial state for the generator circuit
        init_dist = NormalDistribution(int(sum(num_qubits)), mu=1., sigma=1.,
                                       low=bounds[0], high=bounds[1])
        init_distribution = np.sqrt(init_dist.probabilities)
        init_distribution = Custom(num_qubits=sum(num_qubits),
                                   state_vector=init_distribution)
        var_form = RY(int(np.sum(num_qubits)), depth=1,
                      initial_state=init_distribution,
                      entangler_map=entangler_map, entanglement_gate='cz')
        if use_circuits:
            theta = ParameterVector('θ', var_form.num_parameters)
            var_form = var_form.construct_circuit(theta)

        uncertainty_model = UnivariateVariationalDistribution(
            int(sum(num_qubits)), var_form, g_params,
            low=bounds[0], high=bounds[1])

        if use_circuits:
            uncertainty_model._var_form_params = theta

        strike_price = 2
        c_approx = 0.25
        european_call = EuropeanCallExpectedValue(uncertainty_model,
                                                  strike_price=strike_price,
                                                  c_approx=c_approx)

        uncertainty_model.set_probabilities(
            QuantumInstance(BasicAer.get_backend('statevector_simulator')))

        algo = AmplitudeEstimation(5, european_call)
        result = algo.run(quantum_instance=BasicAer.get_backend('statevector_simulator'))
        self.assertAlmostEqual(result['estimation'], 1.2580, places=4)
        self.assertAlmostEqual(result['max_probability'], 0.8785, places=4)

        if not use_circuits:
            warnings.filterwarnings(action="always", category=DeprecationWarning)


if __name__ == '__main__':
    unittest.main()
