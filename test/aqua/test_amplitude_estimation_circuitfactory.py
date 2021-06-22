# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the quantum amplitude estimation algorithm."""

import warnings
import unittest
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, idata, data, unpack
from qiskit import QuantumRegister, QuantumCircuit, BasicAer, execute
from qiskit.circuit.library import QFT
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.uncertainty_models import GaussianConditionalIndependenceModel as GCI
from qiskit.aqua.components.uncertainty_problems import \
    UnivariatePiecewiseLinearObjective as PwlObjective
from qiskit.aqua.components.uncertainty_problems import (MultivariateProblem,
                                                         UncertaintyProblem)
from qiskit.aqua.circuits import WeightedSumOperator
from qiskit.aqua.algorithms import (AmplitudeEstimation, MaximumLikelihoodAmplitudeEstimation,
                                    IterativeAmplitudeEstimation)
from qiskit.aqua.algorithms.amplitude_estimators.q_factory import QFactory


class BernoulliAFactory(UncertaintyProblem):
    r"""Circuit Factory representing the operator A in a Bernoulli problem.

    Given a probability $p$, the operator A prepares the state $\sqrt{1 - p}|0> + \sqrt{p}|1>$.
    """

    def __init__(self, probability=0.5):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
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
    """Circuit Factory representing the operator Q in a Bernoulli problem.

    This implementation exploits the fact that powers of Q can be implemented efficiently by just
    multiplying the angle. Note, that since amplitude estimation only requires controlled powers of
    Q only that method is overridden.
    """

    def __init__(self, bernoulli_expected_value):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
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


class SineIntegralAFactory(UncertaintyProblem):
    r"""Construct the A operator to approximate the integral

        \int_0^1 \sin^2(x) d x

    with a specified number of qubits.
    """

    def __init__(self, num_qubits):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            super().__init__(num_qubits + 1)
        self._i_objective = num_qubits

    def build(self, qc, q, q_ancillas=None, params=None):
        n = self.num_target_qubits - 1
        q_state = [q[i] for i in range(self.num_target_qubits) if i != self._i_objective]
        q_objective = q[self._i_objective]

        # prepare 1/sqrt{2^n} sum_x |x>_n
        for q_i in q_state:
            qc.h(q_i)

        # apply the sine/cosine term
        qc.ry(2 * 1 / 2 / 2**n, q_objective)

        for i, q_i in enumerate(q_state):
            qc.cry(2 * 2**i / 2**n, q_i, q_objective)


@ddt
class TestBernoulli(QiskitAquaTestCase):
    """Tests based on the Bernoulli A operator.

    This class tests
        * the estimation result
        * the constructed circuits
    """

    def setUp(self):
        super().setUp()
        warnings.filterwarnings(action="ignore", category=DeprecationWarning)

        self._statevector = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                            seed_simulator=2, seed_transpiler=2)

        self._unitary = QuantumInstance(backend=BasicAer.get_backend('unitary_simulator'), shots=1,
                                        seed_simulator=42, seed_transpiler=91)

        def qasm(shots=100):
            return QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'), shots=shots,
                                   seed_simulator=2, seed_transpiler=2)

        self._qasm = qasm

    def tearDown(self):
        super().tearDown()
        warnings.filterwarnings(action="always", category=DeprecationWarning)

    @idata([
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
    @unpack
    def test_statevector(self, prob, qae, expect):
        """Test running QAE using the statevector simulator."""
        # construct factories for A and Q
        qae.a_factory = BernoulliAFactory(prob)
        qae.q_factory = BernoulliQFactory(qae.a_factory)

        result = qae.run(self._statevector)

        for key, value in expect.items():
            self.assertAlmostEqual(value, result[key], places=3,
                                   msg="estimate `{}` failed".format(key))

    @idata([
        [0.2, 100, AmplitudeEstimation(4), {'estimation': 0.14644, 'mle': 0.193888}],
        [0.0, 1000, AmplitudeEstimation(2), {'estimation': 0.0, 'mle': 0.0}],
        [0.8, 10, AmplitudeEstimation(7), {'estimation': 0.79784, 'mle': 0.801612}],
        [0.2, 100, MaximumLikelihoodAmplitudeEstimation(4), {'estimation': 0.199606}],
        [0.4, 1000, MaximumLikelihoodAmplitudeEstimation(6), {'estimation': 0.399488}],
        # [0.8, 10, MaximumLikelihoodAmplitudeEstimation(7), {'estimation': 0.800926}],
        [0.2, 100, IterativeAmplitudeEstimation(0.0001, 0.01), {'estimation': 0.199987}],
        [0.4, 1000, IterativeAmplitudeEstimation(0.001, 0.05), {'estimation': 0.400071}],
        [0.8, 10, IterativeAmplitudeEstimation(0.1, 0.05), {'estimation': 0.811711}]
    ])
    @unpack
    def test_qasm(self, prob, shots, qae, expect):
        """ qasm test """
        # construct factories for A and Q
        qae.a_factory = BernoulliAFactory(prob)
        qae.q_factory = BernoulliQFactory(qae.a_factory)

        result = qae.run(self._qasm(shots))

        for key, value in expect.items():
            self.assertAlmostEqual(value, result[key], places=3,
                                   msg="estimate `{}` failed".format(key))

    @data(True, False)
    def test_qae_circuit(self, efficient_circuit):
        """Test circuits resulting from canonical amplitude estimation.

        Build the circuit manually and from the algorithm and compare the resulting unitaries.
        """

        prob = 0.5

        for m in range(2, 7):
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            qae = AmplitudeEstimation(m, a_factory=BernoulliAFactory(prob))
            angle = 2 * np.arcsin(np.sqrt(prob))

            # manually set up the inefficient AE circuit
            q_ancilla = QuantumRegister(m, 'a')
            q_objective = QuantumRegister(1, 'q')
            circuit = QuantumCircuit(q_ancilla, q_objective)

            # initial Hadamard gates
            for i in range(m):
                circuit.h(q_ancilla[i])

            # A operator
            circuit.ry(angle, q_objective)

            if efficient_circuit:
                qae.q_factory = BernoulliQFactory(qae.a_factory)
                for power in range(m):
                    circuit.cry(2 * 2 ** power * angle, q_ancilla[power], q_objective[0])

            else:
                q_factory = QFactory(qae.a_factory, i_objective=0)
                for power in range(m):
                    for _ in range(2**power):
                        q_factory.build_controlled(circuit, q_objective, q_ancilla[power])

            warnings.filterwarnings('always', category=DeprecationWarning)
            # fourier transform
            iqft = QFT(m, do_swaps=False).inverse().reverse_bits()
            circuit.append(iqft.to_instruction(), q_ancilla)

            expected_unitary = self._unitary.execute(circuit).get_unitary()

            actual_circuit = qae.construct_circuit(measurement=False)
            actual_unitary = self._unitary.execute(actual_circuit).get_unitary()

            diff = np.sum(np.abs(actual_unitary - expected_unitary))
            self.assertAlmostEqual(diff, 0)

    @data(True, False)
    def test_iqae_circuits(self, efficient_circuit):
        """Test circuits resulting from iterative amplitude estimation.

        Build the circuit manually and from the algorithm and compare the resulting unitaries.
        """
        prob = 0.5

        for k in range(2, 7):
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            qae = IterativeAmplitudeEstimation(0.01, 0.05, a_factory=BernoulliAFactory(prob))
            angle = 2 * np.arcsin(np.sqrt(prob))

            # manually set up the inefficient AE circuit
            q_objective = QuantumRegister(1, 'q')
            circuit = QuantumCircuit(q_objective)

            # A operator
            circuit.ry(angle, q_objective)

            if efficient_circuit:
                qae.q_factory = BernoulliQFactory(qae.a_factory)
                # for power in range(k):
                #    circuit.ry(2 ** power * angle, q_objective[0])
                circuit.ry(2 * k * angle, q_objective[0])

            else:
                q_factory = QFactory(qae.a_factory, i_objective=0)
                for _ in range(k):
                    q_factory.build(circuit, q_objective)
            warnings.filterwarnings('always', category=DeprecationWarning)

            expected_unitary = self._unitary.execute(circuit).get_unitary()

            actual_circuit = qae.construct_circuit(k, measurement=False)
            actual_unitary = self._unitary.execute(actual_circuit).get_unitary()

            diff = np.sum(np.abs(actual_unitary - expected_unitary))
            self.assertAlmostEqual(diff, 0)

    @data(True, False)
    def test_mlae_circuits(self, efficient_circuit):
        """ Test the circuits constructed for MLAE """
        prob = 0.5

        for k in range(1, 7):
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            qae = MaximumLikelihoodAmplitudeEstimation(k, a_factory=BernoulliAFactory(prob))
            angle = 2 * np.arcsin(np.sqrt(prob))

            # compute all the circuits used for MLAE
            circuits = []

            # 0th power
            q_objective = QuantumRegister(1, 'q')
            circuit = QuantumCircuit(q_objective)
            circuit.ry(angle, q_objective)
            circuits += [circuit]

            # powers of 2
            for power in range(k):
                q_objective = QuantumRegister(1, 'q')
                circuit = QuantumCircuit(q_objective)

                # A operator
                circuit.ry(angle, q_objective)

                # Q^(2^j) operator
                if efficient_circuit:
                    qae.q_factory = BernoulliQFactory(qae.a_factory)
                    circuit.ry(2 * 2 ** power * angle, q_objective[0])

                else:
                    q_factory = QFactory(qae.a_factory, i_objective=0)
                    for _ in range(2**power):
                        q_factory.build(circuit, q_objective)

            warnings.filterwarnings('always', category=DeprecationWarning)
            actual_circuits = qae.construct_circuits(measurement=False)

            for actual, expected in zip(actual_circuits, circuits):
                expected_unitary = self._unitary.execute(expected).get_unitary()
                actual_unitary = self._unitary.execute(actual).get_unitary()
                diff = np.sum(np.abs(actual_unitary - expected_unitary))
                self.assertAlmostEqual(diff, 0)


@ddt
class TestProblemSetting(QiskitAquaTestCase):
    """Test the setting and getting of the A and Q operator and the objective qubit index."""

    def setUp(self):
        super().setUp()
        self.a_bernoulli = BernoulliAFactory(0)
        self.q_bernoulli = BernoulliQFactory(self.a_bernoulli)
        self.i_bernoulli = 0

        num_qubits = 5
        self.a_integral = SineIntegralAFactory(num_qubits)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            self.q_integral = QFactory(self.a_integral, num_qubits)
        self.i_integral = num_qubits

    @idata([
        [AmplitudeEstimation(2)],
        [IterativeAmplitudeEstimation(0.1, 0.001)],
        [MaximumLikelihoodAmplitudeEstimation(3)],
    ])
    @unpack
    def test_operators(self, qae):
        """ Test if A/Q operator + i_objective set correctly """
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        self.assertIsNone(qae.a_factory)
        self.assertIsNone(qae.q_factory)
        self.assertIsNone(qae.i_objective)
        self.assertIsNone(qae._a_factory)
        self.assertIsNone(qae._q_factory)
        self.assertIsNone(qae._i_objective)

        qae.a_factory = self.a_bernoulli
        self.assertIsNotNone(qae.a_factory)
        self.assertIsNotNone(qae.q_factory)
        self.assertIsNotNone(qae.i_objective)
        self.assertIsNotNone(qae._a_factory)
        self.assertIsNone(qae._q_factory)
        self.assertIsNone(qae._i_objective)

        qae.q_factory = self.q_bernoulli
        self.assertIsNotNone(qae.a_factory)
        self.assertIsNotNone(qae.q_factory)
        self.assertIsNotNone(qae.i_objective)
        self.assertIsNotNone(qae._a_factory)
        self.assertIsNotNone(qae._q_factory)
        self.assertIsNone(qae._i_objective)

        qae.i_objective = self.i_bernoulli
        self.assertIsNotNone(qae.a_factory)
        self.assertIsNotNone(qae.q_factory)
        self.assertIsNotNone(qae.i_objective)
        self.assertIsNotNone(qae._a_factory)
        self.assertIsNotNone(qae._q_factory)
        self.assertIsNotNone(qae._i_objective)
        warnings.filterwarnings('always', category=DeprecationWarning)

    @data(
        AmplitudeEstimation(2),
        IterativeAmplitudeEstimation(0.1, 0.001),
        MaximumLikelihoodAmplitudeEstimation(3),
    )
    def test_a_factory_update(self, qae):
        """Test if the Q factory is updated if the a_factory changes -- except set manually."""
        # Case 1: Set to BernoulliAFactory with default Q operator
        warnings.filterwarnings(action="ignore", category=DeprecationWarning)
        qae.a_factory = self.a_bernoulli
        self.assertIsInstance(qae.q_factory.a_factory, BernoulliAFactory)
        self.assertEqual(qae.i_objective, self.i_bernoulli)

        # Case 2: Change to SineIntegralAFactory with default Q operator
        qae.a_factory = self.a_integral
        self.assertIsInstance(qae.q_factory.a_factory, SineIntegralAFactory)
        self.assertEqual(qae.i_objective, self.i_integral)

        # Case 3: Set to BernoulliAFactory with special Q operator
        qae.a_factory = self.a_bernoulli
        qae.q_factory = self.q_bernoulli
        self.assertIsInstance(qae.q_factory, BernoulliQFactory)
        self.assertEqual(qae.i_objective, self.i_bernoulli)

        # Case 4: Set to SineIntegralAFactory, and do not set Q. Then the old Q operator
        # should remain
        qae.a_factory = self.a_integral
        self.assertIsInstance(qae.q_factory, BernoulliQFactory)
        self.assertEqual(qae.i_objective, self.i_bernoulli)
        warnings.filterwarnings(action="always", category=DeprecationWarning)


@ddt
class TestSineIntegral(QiskitAquaTestCase):
    """Tests based on the A operator to integrate sin^2(x).

    This class tests
        * the estimation result
        * the confidence intervals
    """

    def setUp(self):
        super().setUp()
        self._statevector = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                            seed_simulator=123,
                                            seed_transpiler=41)

        def qasm(shots=100):
            return QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'), shots=shots,
                                   seed_simulator=7192, seed_transpiler=90000)

        self._qasm = qasm

    @idata([
        [2, AmplitudeEstimation(2), {'estimation': 0.5, 'mle': 0.270290}],
        [4, MaximumLikelihoodAmplitudeEstimation(4), {'estimation': 0.272675}],
        [3, IterativeAmplitudeEstimation(0.1, 0.1), {'estimation': 0.272082}],
    ])
    @unpack
    def test_statevector(self, n, qae, expect):
        """ Statevector end-to-end test """
        # construct factories for A and Q
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            qae.a_factory = SineIntegralAFactory(n)

        result = qae.run(self._statevector)

        for key, value in expect.items():
            self.assertAlmostEqual(value, result[key], places=3,
                                   msg="estimate `{}` failed".format(key))

    @idata([
        [4, 10, AmplitudeEstimation(2), {'estimation': 0.5, 'mle': 0.333333}],
        [3, 10, MaximumLikelihoodAmplitudeEstimation(2), {'estimation': 0.256878}],
        [3, 1000, IterativeAmplitudeEstimation(0.01, 0.01), {'estimation': 0.271790}],
    ])
    @unpack
    def test_qasm(self, n, shots, qae, expect):
        """QASM simulator end-to-end test."""
        # construct factories for A and Q
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            qae.a_factory = SineIntegralAFactory(n)

        result = qae.run(self._qasm(shots))

        for key, value in expect.items():
            self.assertAlmostEqual(value, result[key], places=3,
                                   msg="estimate `{}` failed".format(key))

    @idata([
        [AmplitudeEstimation(3), 'mle',
         {'likelihood_ratio': [0.24947346406470136, 0.3003771197734433],
          'fisher': [0.24861769995820207, 0.2999286066724035],
          'observed_fisher': [0.24845622030041542, 0.30009008633019013]}
         ],
        [MaximumLikelihoodAmplitudeEstimation(3), 'estimation',
         {'likelihood_ratio': [0.25987941798909114, 0.27985361366769945],
          'fisher': [0.2584889015125656, 0.2797018754936686],
          'observed_fisher': [0.2659279996107888, 0.2722627773954454]}],
    ])
    @unpack
    def test_confidence_intervals(self, qae, key, expect):
        """End-to-end test for all confidence intervals."""
        n = 3
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            qae.a_factory = SineIntegralAFactory(n)

        # statevector simulator
        result = qae.run(self._statevector)
        methods = ['lr', 'fi', 'oi']  # short for likelihood_ratio, fisher, observed_fisher
        alphas = [0.1, 0.00001, 0.9]  # alpha shouldn't matter in statevector
        for alpha, method in zip(alphas, methods):
            confint = qae.confidence_interval(alpha, method)
            # confidence interval based on statevector should be empty, as we are sure of the result
            self.assertAlmostEqual(confint[1] - confint[0], 0.0)
            self.assertAlmostEqual(confint[0], result[key])

        # qasm simulator
        shots = 100
        alpha = 0.01
        result = qae.run(self._qasm(shots))
        for method, expected_confint in expect.items():
            confint = qae.confidence_interval(alpha, method)
            np.testing.assert_almost_equal(confint, expected_confint, decimal=10)
            self.assertTrue(confint[0] <= result[key] <= confint[1])

    def test_iqae_confidence_intervals(self):
        """End-to-end test for the IQAE confidence interval."""
        n = 3
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            qae = IterativeAmplitudeEstimation(0.1, 0.01, a_factory=SineIntegralAFactory(n))
        expected_confint = [0.19840508760087738, 0.35110155403424115]

        # statevector simulator
        result = qae.run(self._statevector)
        confint = result['confidence_interval']
        # confidence interval based on statevector should be empty, as we are sure of the result
        self.assertAlmostEqual(confint[1] - confint[0], 0.0)
        self.assertAlmostEqual(confint[0], result['estimation'])

        # qasm simulator
        shots = 100
        result = qae.run(self._qasm(shots))
        confint = result['confidence_interval']
        np.testing.assert_almost_equal(confint, expected_confint, decimal=7)
        self.assertTrue(confint[0] <= result['estimation'] <= confint[1])


@ddt
class TestCreditRiskAnalysis(QiskitAquaTestCase):
    """Test a more difficult example, motivated from Credit Risk Analysis."""

    def setUp(self):
        super().setUp()
        warnings.filterwarnings(action="ignore", category=DeprecationWarning)

    def tearDown(self):
        super().tearDown()
        warnings.filterwarnings(action="always", category=DeprecationWarning)

    @data('statevector_simulator')
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
