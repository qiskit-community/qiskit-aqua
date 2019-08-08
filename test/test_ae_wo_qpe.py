import unittest
from parameterized import parameterized
import numpy as np

from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import AmplitudeEstimationWithoutQPE
from qiskit.aqua.algorithms.single_sample.amplitude_estimation.q_factory import QFactory
from qiskit.aqua.components.uncertainty_problems import UncertaintyProblem

from test.common import QiskitAquaTestCase


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


class TestAEwoQPE(QiskitAquaTestCase):
    """
    Test the Amplitude Estimation algorithms.
    """

    def setUp(self):
        super().setUp()

        self._statevector = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                            circuit_caching=False, seed_simulator=2, seed_transpiler=2)
        self._qasm = QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'), shots=1000,
                                     circuit_caching=False, seed_simulator=2, seed_transpiler=2)

    @parameterized.expand([
        [0.2, 2],
        [0.4, 4],
        [0.82, 5],
        [0.49, 3],
        [0.5, 2]
    ])
    def test_statevector(self, p, log_max_evals):
        np.random.seed(1)

        # construct factories for A and Q
        a_factory = BernoulliAFactory(p)
        q_factory = BernoulliQFactory(a_factory)

        ae = AmplitudeEstimationWithoutQPE(log_max_evals, a_factory, i_objective=0, q_factory=q_factory)
        result = ae.run(self._statevector)

        self.assertAlmostEqual(result['estimation'], p, places=5)

    @parameterized.expand([
        [0.2, 4, 0.199606, 100],
        [0.4, 6, 0.399488, 1000],
        [0.8, 7, 0.799940, 10]
    ])
    def test_qasm(self, p, log_max_evals, expect, shots):
        # construct factories for A and Q
        a_factory = BernoulliAFactory(p)
        q_factory = BernoulliQFactory(a_factory)

        ae = AmplitudeEstimationWithoutQPE(log_max_evals, a_factory, i_objective=0, q_factory=q_factory)
        result = ae.run(self._qasm)

        self.assertAlmostEqual(result['estimation'], expect, places=5)

    @parameterized.expand([
        [0.2, 4, (0.19917, 0.20003), 100, "observed_fisher"],
        [0.4, 6, (0.39908, 0.39988), 1000, "fisher"],
        [0.8, 7, (0.79983, 0.80008), 10, "likelihood_ratio"]
    ])
    def test_ci(self, p, log_max_evals, expect, shots, kind):
        alpha = 0.05

        # construct factories for A and Q
        a_factory = BernoulliAFactory(p)
        q_factory = BernoulliQFactory(a_factory)

        ae = AmplitudeEstimationWithoutQPE(log_max_evals, a_factory, i_objective=0, q_factory=q_factory)
        ae.run(self._qasm)
        ci = ae.confidence_interval(alpha, kind=kind)

        self.assertAlmostEqual(ci[0], expect[0], places=3)
        self.assertAlmostEqual(ci[1], expect[1], places=3)


if __name__ == "__main__":
    unittest.main()
