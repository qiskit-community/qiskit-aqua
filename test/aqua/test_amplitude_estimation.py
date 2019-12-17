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

""" Test Amplitude Estimation """

import unittest
from test.aqua.common import QiskitAquaTestCase
import numpy as np
from parameterized import parameterized
from qiskit import QuantumRegister, QuantumCircuit, BasicAer, execute
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.uncertainty_models import GaussianConditionalIndependenceModel as GCI
from qiskit.aqua.components.uncertainty_problems import \
    UnivariatePiecewiseLinearObjective as PwlObjective
from qiskit.aqua.components.uncertainty_problems import (MultivariateProblem,
                                                         UncertaintyProblem)
from qiskit.aqua.circuits import WeightedSumOperator
from qiskit.aqua.algorithms import (AmplitudeEstimation, MaximumLikelihoodAmplitudeEstimation,
                                    IterativeAmplitudeEstimation)
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

    def build(self, qc, q, q_ancillas=None, params=None):
        # A is a rotation of angle theta_p around the Y-axis
        qc.ry(self._theta_p, q[self.i_state])

    def value_to_estimation(self, value):
        return value


class BernoulliQFactory(QFactory):
    """
    Circuit Factory representing the operator Q.
    This implementation exploits the fact that powers of Q
    can be implemented efficiently by just multiplying the angle.
    (amplitude estimation only requires controlled powers of Q,
    thus, only this method is overridden.)
    """

    def __init__(self, bernoulli_expected_value):
        super().__init__(bernoulli_expected_value, i_objective=0)

    def build(self, qc, q, q_ancillas=None, params=None):
        i_state = self.a_factory.i_state
        theta_p = self.a_factory._theta_p
        # Q is a rotation of angle 2*theta_p around the Y-axis
        qc.ry(2 * theta_p, q[i_state])

    def build_power(self, qc, q, power, q_ancillas=None):
        i_state = self.a_factory.i_state
        theta_p = self.a_factory._theta_p
        qc.ry(2 * power * theta_p, q[i_state])

    def build_controlled_power(self, qc, q, q_control, power,
                               q_ancillas=None, use_basis_gates=True):
        i_state = self.a_factory.i_state
        theta_p = self.a_factory._theta_p
        qc.cry(2 * power * theta_p, q_control, q[i_state])


class TestBernoulli(QiskitAquaTestCase):
    """ Test Bernoulli """

    def setUp(self):
        super().setUp()

        self._statevector = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                            seed_simulator=2,
                                            seed_transpiler=2)

        def qasm(shots=100):
            return QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'), shots=shots,
                                   seed_simulator=2, seed_transpiler=2)

        self._qasm = qasm

    @parameterized.expand([
        [0.2, AmplitudeEstimation(2), {'estimation': 0.5, 'mle': 0.2}],
        [0.4, AmplitudeEstimation(4), {'estimation': 0.30866, 'mle': 0.4}],
        [0.82, AmplitudeEstimation(5), {'estimation': 0.85355, 'mle': 0.82}],
        [0.49, AmplitudeEstimation(3), {'estimation': 0.5, 'mle': 0.49}],
        [0.2, MaximumLikelihoodAmplitudeEstimation(2), {'estimation': 0.2}],
        [0.4, MaximumLikelihoodAmplitudeEstimation(4), {'estimation': 0.4}],
        [0.82, MaximumLikelihoodAmplitudeEstimation(5), {'estimation': 0.82}],
        [0.49, MaximumLikelihoodAmplitudeEstimation(3), {'estimation': 0.49}],
        [0.2, IterativeAmplitudeEstimation(0.1, 0.1), {'estimation': 0.2}],
        [0.4, IterativeAmplitudeEstimation(0.00001, 0.01), {'estimation': 0.4}],
        [0.82, IterativeAmplitudeEstimation(0.00001, 0.05), {'estimation': 0.82}],
        [0.49, IterativeAmplitudeEstimation(0.001, 0.01), {'estimation': 0.49}]
    ])
    def test_statevector(self, prob, a_e, expect):
        """ statevector test """
        # construct factories for A and Q
        a_e.a_factory = BernoulliAFactory(prob)
        a_e.q_factory = BernoulliQFactory(a_e.a_factory)

        result = a_e.run(self._statevector)

        for key, value in expect.items():
            self.assertAlmostEqual(value, result[key], places=3,
                                   msg="estimate `{}` failed".format(key))

    @parameterized.expand([
        [0.2, 100, AmplitudeEstimation(4), {'estimation': 0.14644, 'mle': 0.193888}],
        [0.0, 1000, AmplitudeEstimation(2), {'estimation': 0.0, 'mle': 0.0}],
        [0.8, 10, AmplitudeEstimation(7), {'estimation': 0.79784, 'mle': 0.801612}],
        [0.2, 100, MaximumLikelihoodAmplitudeEstimation(4), {'estimation': 0.199606}],
        [0.4, 1000, MaximumLikelihoodAmplitudeEstimation(6), {'estimation': 0.399488}],
        [0.8, 10, MaximumLikelihoodAmplitudeEstimation(7), {'estimation': 0.800926}],
        [0.2, 100, IterativeAmplitudeEstimation(0.0001, 0.01), {'estimation': 0.199987}],
        [0.4, 1000, IterativeAmplitudeEstimation(0.001, 0.05), {'estimation': 0.400071}],
        [0.8, 10, IterativeAmplitudeEstimation(0.1, 0.05), {'estimation': 0.811711}]
    ])
    def test_qasm(self, prob, shots, a_e, expect):
        """ qasm test """
        # construct factories for A and Q
        a_e.a_factory = BernoulliAFactory(prob)
        a_e.q_factory = BernoulliQFactory(a_e.a_factory)

        result = a_e.run(self._qasm(shots))

        for key, value in expect.items():
            self.assertAlmostEqual(value, result[key], places=3,
                                   msg="estimate `{}` failed".format(key))


class TestCreditRiskAnalysis(QiskitAquaTestCase):
    """ Test Credit Risk Analysis """
    @parameterized.expand([
        'statevector_simulator'
    ])
    def test_conditional_value_at_risk(self, simulator):
        """ conditional value at risk test """
        # define backend to be used
        backend = BasicAer.get_backend(simulator)

        # set problem parameters
        n_z = 2
        z_max = 2
        # z_values = np.linspace(-z_max, z_max, 2 ** n_z)
        p_zeros = [0.15, 0.25]
        rhos = [0.1, 0.05]
        lgd = [1, 2]
        k_l = len(p_zeros)
        # alpha = 0.05

        # set var value
        var = 2
        var_prob = 0.961940

        # determine number of qubits required to represent total loss
        # n_s = WeightedSumOperator.get_required_sum_qubits(lgd)

        # create circuit factory (add Z qubits with weight/loss 0)
        agg = WeightedSumOperator(n_z + k_l, [0] * n_z + lgd)

        # define linear objective
        breakpoints = [0, var]
        slopes = [0, 1]
        offsets = [0, 0]  # subtract VaR and add it later to the estimate
        f_min = 0
        f_max = 3 - var
        c_approx = 0.25

        # construct circuit factory for uncertainty model (Gaussian Conditional Independence model)
        gci = GCI(n_z, z_max, p_zeros, rhos)

        cvar_objective = PwlObjective(
            agg.num_sum_qubits,
            0,
            2 ** agg.num_sum_qubits - 1,  # max value that can be reached by the qubit register
                                          # (will not always be reached)
            breakpoints,
            slopes,
            offsets,
            f_min,
            f_max,
            c_approx
        )

        multivariate_cvar = MultivariateProblem(gci, agg, cvar_objective)

        num_qubits = multivariate_cvar.num_target_qubits
        num_ancillas = multivariate_cvar.required_ancillas()

        q = QuantumRegister(num_qubits, name='q')
        q_a = QuantumRegister(num_ancillas, name='q_a')
        qc = QuantumCircuit(q, q_a)

        multivariate_cvar.build(qc, q, q_a)

        job = execute(qc, backend=backend)

        # evaluate resulting statevector
        value = 0
        for i, a_i in enumerate(job.result().get_statevector()):
            b = ('{0:0%sb}' %
                 multivariate_cvar.num_target_qubits).\
                format(i)[-multivariate_cvar.num_target_qubits:]
            a_m = np.round(np.real(a_i), decimals=4)
            if np.abs(a_m) > 1e-6 and b[0] == '1':
                value += a_m ** 2

        # normalize and add VaR to estimate
        value = multivariate_cvar.value_to_estimation(value)
        normalized_value = value / (1.0 - var_prob) + var

        # compare to precomputed solution
        self.assertEqual(0.0, np.round(normalized_value - 3.3796, decimals=4))


if __name__ == '__main__':
    unittest.main()
