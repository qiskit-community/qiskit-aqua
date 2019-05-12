# TODO
# (c) 2019 IBM

"""
An iterative version of the Amplitude Estimation algorithm based on
https://arxiv.org/pdf/1904.10246.pdf
"""

import logging
# from collections import OrderedDict
import numpy as np

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.aqua import AquaError
# from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.algorithms.single_sample.amplitude_estimation.q_factory import QFactory
from qiskit.aqua.algorithms.single_sample.amplitude_estimation.ci_utils import chi2_quantile

logger = logging.getLogger(__name__)


class IterativeAmplitudeEstimation(QuantumAlgorithm):

    CONFIGURATION = {
        'name': 'IterativeAmplitudeEstimation',
        'description': 'Iterative Amplitude Estimation Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'AmplitudeEstimation_schema',
            'type': 'object',
            'properties': {
                'num_eval_qubits': {
                    'type': 'integer',
                    'default': 5,
                    'minimum': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['uncertainty'],
        'depends': [
            {
                'pluggable_type': 'uncertainty_problem',
                'default': {
                    'name': 'EuropeanCallDelta'
                }
            },
        ],
    }

    def __init__(self, num_iterations, a_factory,
                 q_factory=None, rotations=None):
        """
        Constructor.

        Args:
            num_iterations (int): number of iterations in the algorithm
            a_factory (CircuitFactory): the CircuitFactory subclass object
                                        representing the problem unitary
            q_factory (CircuitFactory): the CircuitFactory subclass object
                                        representing an amplitude estimation
                                        sample (based on a_factory)
            rotations (iterable of ints): The number of times the operator Q
                                          is applied in each iteration,
                                          overwrites num_iterations
        """
        super().__init__()

        # get/construct A/Q operator
        self.a_factory = a_factory
        if q_factory is None:
            self.q_factory = QFactory(a_factory)
        else:
            self.q_factory = q_factory

        # get parameters
        self._num_iterations = num_iterations
        if rotations is None:
            self._rotations = 2**np.arange(num_iterations)
        else:
            self._rotations = rotations

        # Store likelihood functions of single experiments here
        self._likelihoods = []

        self._ret = {}

        # determine number of ancillas
        self._num_ancillas = self.q_factory.required_ancillas_controlled()
        self._num_qubits = self.a_factory.num_target_qubits + \
            + self._num_ancillas

        print("ancillas:", self._num_ancillas)
        print("targets:", self.a_factory.num_target_qubits)
        print("num_qubits", self._num_qubits)

    def get_single_likelihood(self, good, total, num_rotations):
        """
        @brief
        """
        def likelihood(theta):
            return (np.sin((2 * num_rotations + 1) * theta)**2)**good \
                * (np.cos((2 * num_rotations + 1) * theta)**2)**(total - good)
        return likelihood

    def construct_single_circuit(self, num_rotations, measurement=False):
        q = QuantumRegister(self.q_factory.num_target_qubits, name='q')
        self._state_register = q

        qc = QuantumCircuit(q)

        num_aux_qubits, aux = 0, None
        if self.a_factory is not None:
            num_aux_qubits = self.a_factory.required_ancillas()
        if self.q_factory is not None:
            num_aux_qubits = max(num_aux_qubits,
                                 self.q_factory.required_ancillas_controlled())

        if num_aux_qubits > 0:
            aux = QuantumRegister(num_aux_qubits, name='aux')
            qc.add_register(aux)

        print("Final circ size:", len(qc.qubits))

        self.a_factory.build(qc, q, aux)
        self.q_factory.build_power(qc, q, num_rotations, aux)

        if measurement:
            q_ancilla = ClassicalRegister(self.q_factory.num_target_qubits,
                                          name='qa')
            qc.add_register(q_ancilla)
            # qc.barrier(a)
            qc.measure(q, q_ancilla)

        return qc

    def maximize(self, likelihood):
        # Should be this many numbers also for LR statistic later in self.ci!
        thetas = np.linspace(0, np.pi / 2, num=int(1e6))
        vals = np.array([likelihood(t) for t in thetas])

        # Avoid double evaluation in likelihood ratio
        self._thetas_grid = thetas
        self._logliks_grid = np.log(vals)

        idx = np.argmax(vals)

        return thetas[idx]

    def get_mle(self):
        if len(self._likelihoods) == 0:
            raise AquaError("likelihoods empty, call the method `run` first")

        def likelihood(theta):
            return np.prod([lik(theta) for lik in self._likelihoods])

        return self.maximize(likelihood)

    def ci(self, alpha, kind="likelihood_ratio"):
        if kind == "likelihood_ratio":
            # Threshold defining confidence interval
            loglik_mle = np.max(self._logliks_grid)
            thres = loglik_mle - chi2_quantile(alpha) / 2

            # Which values are above the threshold?
            above_thres = self._thetas_grid[self._logliks_grid >= thres]

            # Get boundaries
            # since thetas_grid is sorted [0] == min, [1] == max
            ci = np.array([above_thres[0], above_thres[-1]])

            return ci
        else:
            raise AquaError(f"confidence interval kind {kind} not implemented")

    def _run(self):
        for num_rotations in self._rotations:
            if self._quantum_instance.is_statevector:
                qc = self.construct_single_circuit(num_rotations,
                                                   measurement=False)
                # run circuit on statevector simlator
                ret = self._quantum_instance.execute(qc)
                state_vector = np.asarray([ret.get_statevector(qc)])
                self._ret['statevector'] = state_vector

                print(state_vector)
                pr = np.real(state_vector.conj() * state_vector)
                print(pr)
                print(pr.shape)

                # get probability for good measurement
                pr_good = np.real(state_vector.conj() * state_vector).flatten()[1]

                good_counts = pr_good
                total_counts = 1

            else:
                # run circuit on QASM simulator
                qc = self.construct_single_circuit(num_rotations,
                                                   measurement=True)
                ret = self._quantum_instance.execute(qc)

                # get counts
                self._ret['counts'] = ret.get_counts()

                # sum all counts where last qubit is one
                good_counts = ret.get_counts()['1']
                total_counts = sum(ret.get_counts().values())

            self._likelihoods.append(self.get_single_likelihood(good_counts,
                                                                total_counts,
                                                                num_rotations))

        self._ret['angle'] = self.get_mle()
        self._ret['value'] = np.sin(self._ret['angle'])**2
        self._ret['estimation'] = self.a_factory.value_to_estimation(
            self._ret['value'])

        return self._ret
