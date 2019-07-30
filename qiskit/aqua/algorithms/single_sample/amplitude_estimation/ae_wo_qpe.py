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
"""
The Amplitude Estimation Algorithm.
"""

import logging
from collections import OrderedDict
import numpy as np
from scipy.stats import norm, chi2

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.aqua import AquaError
from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.algorithms import QuantumAlgorithm

from .q_factory import QFactory

logger = logging.getLogger(__name__)


class AmplitudeEstimationWithoutQPE(QuantumAlgorithm):
    """
    The Amplitude Estimation without QPE algorithm.
    """

    CONFIGURATION = {
        'name': 'AmplitudeEstimationWithoutQPE',
        'description': 'Amplitude Estimation Without QPE Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'AmplitudeEstimationWithoutQPE_schema',
            'type': 'object',
            'properties': {
                'log_max_evals': {
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

    def __init__(self, log_max_evals, a_factory, i_objective=None, q_factory=None):
        """
        Constructor.

        Args:
            log_max_evals (int): base-2-logarithm of maximal number of evaluations - resulting evaluation schedule will be [Q^2^0, ..., Q^2^{max_evals_log-1}]
            a_factory (CircuitFactory): the CircuitFactory subclass object representing the problem unitary
            i_objective (int): index of qubit representing the objective in the uncertainty problem
            q_factory (CircuitFactory): the CircuitFactory subclass object representing an amplitude estimation sample (based on a_factory)
        """
        self.validate(locals())
        super().__init__()

        # get/construct A/Q operator
        self.a_factory = a_factory
        if q_factory is None:
            if i_objective is None:
                i_objective = self.a_factory.num_target_qubits - 1
            self.q_factory = QFactory(a_factory, i_objective)
        else:
            if i_objective is None:
                raise AquaError('i_objective must be set for custom q_factory')
            self.q_factory = q_factory
        self.i_objective = i_objective

        # get parameters
        self._log_max_evals = log_max_evals
        self._evaluation_schedule = [2**j for j in range(log_max_evals)]

        # determine number of ancillas
        self._num_ancillas = self.q_factory.required_ancillas()
        self._num_qubits = self.a_factory.num_target_qubits + self._num_ancillas

        self._circuits = []
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: Input instance
        """
        if algo_input is not None:
            raise AquaError("Input instance not supported.")

        ae_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        log_max_evals = ae_params.get('log_max_evals')

        # Set up uncertainty problem. The params can include an uncertainty model
        # type dependent on the uncertainty problem and is this its responsibility
        # to create for itself from the complete params set that is passed to it.
        uncertainty_problem_params = params.get(Pluggable.SECTION_KEY_UNCERTAINTY_PROBLEM)
        uncertainty_problem = get_pluggable_class(
            PluggableType.UNCERTAINTY_PROBLEM,
            uncertainty_problem_params['name']).init_params(params)

        return cls(log_max_evals, uncertainty_problem, q_factory=None)

    def construct_circuits(self, measurement=False):
        """
        Construct the Amplitude Estimation w/o QPE quantum circuits.

        Args:
            measurement (bool): Boolean flag to indicate if measurement should be included in the circuits.

        Returns:
            a list with the QuantumCircuit objects for the algorithm
        """

        # construct first part of circuit
        q = QuantumRegister(self.a_factory.num_target_qubits)
        if measurement:
            c = ClassicalRegister(1)
            qc_a = QuantumCircuit(q, c, name='qc_a')
        else:
            qc_a = QuantumCircuit(q, name='qc_a')
        self.a_factory.build(qc_a, q)

        self._circuits = []
        for k in self._evaluation_schedule:
            qc_k = qc_a.copy(name='qc_a_q_%s' % k)
            self.q_factory.build_power(qc_k, q, k)

            if measurement:
                qc_k.measure(q[self.i_objective], c[0])

            self._circuits += [qc_k]

        return self._circuits

    def _evaluate_statevectors(self, state_vectors):

        probabilities = []
        for sv in state_vectors:
            p_k = 0
            for i, a in enumerate(sv):
                p = np.abs(a)**2
                b = ('{0:%sb}' % self._num_qubits).format(i)[::-1]
                if b[self.i_objective] == '1':
                    p_k += p
            probabilities += [p_k]

        return probabilities

    def _evaluate_counts(self, counts):

        probabilities = []
        for c in counts:
            num_shots = sum(c.values())
            probabilities += [c['1'] / num_shots]

        return probabilities

    def _run_mle(self, probabilities):
        # TODO: replace by more efficient and numerically stable implementation
        def loglikelihood(theta, probs):
            L = 0
            for i, k in enumerate(self._evaluation_schedule):
                L += np.log(np.sin((2 * k + 1) * theta) ** 2) * probs[i]
                L += np.log(np.cos((2 * k + 1) * theta) ** 2) * (1 - probs[i])
            return L

        num_points = 10000
        thetas = np.linspace(np.pi / num_points / 2, np.pi / 2, num_points)
        values = np.zeros(len(thetas))
        for i, t in enumerate(thetas):
            values[i] = loglikelihood(t, probabilities)

        i_max = np.argmax(values)
        return thetas[i_max]

    def compute_lr_ci(self, alpha=0.05, nevals=10000):

        def loglikelihood(theta, one_counts, all_counts):
            L = 0
            for i, k in enumerate(self._evaluation_schedule):
                L += np.log(np.sin((2 * k + 1) * theta) ** 2) * one_counts[i]
                L += np.log(np.cos((2 * k + 1) * theta) ** 2) * (all_counts[i] - one_counts[i])
            return L

        one_counts = []
        all_counts = []
        for c in self._ret['counts']:
            one_counts += c['1']
            all_counts += sum(c.values())

        thetas = np.linspace(np.pi / nevals / 2, np.pi / 2, nevals)
        values = np.zeros(len(thetas))
        for i, t in enumerate(thetas):
            values[i] = self._loglikelihood(t, one_counts, all_counts)

        loglik_mle = loglikelihood(self._ret['theta'], one_counts, all_counts)
        chi2_quantile = 1 - chi2.ppf(1 - alpha)
        thres = loglik_mle - chi2_quantile / 2

        # the outer LR confidence interval
        above_thres = thetas[values >= thres]
        ci_outer = [np.min(above_thres), np.max(above_thres)]
        mapped_ci_outer = [self.a_factory.value_to_estimation(bound) for bound in ci_outer]

        # the inner LR confidence interval:
        # [largest value below mle and above thres, smallest value above mle and above thres]
        larger_than_mle = above_thres[above_thres > self._ret['theta']]
        smaller_than_mle = above_thres[above_thres < self._ret['theta']]
        ci_inner = [np.max(smaller_than_mle), np.min(larger_than_mle)]
        mapped_ci_inner = [self.a_factory.value_to_estimation(bound) for bound in ci_inner]

        return mapped_ci_outer, mapped_ci_inner

    def compute_fisher_ci(self, alpha=0.05):
        normal_quantile = norm.ppf(1 - alpha / 2)
        ci = self._ret['estimation'] + normal_quantile / np.sqrt(self._ret['fisher_information']) * np.array([-1, 1])
        mapped_ci = [self.a_factory.value_to_estimation(bound) for bound in ci]
        return mapped_ci

    def _compute_fisher_information(self):
        # the fisher information is infinite, since:
        # 1) statevector simulation should return the exact value
        # 2) statevector probabilities correspond to "infinite" shots
        if self._quantum_instance.is_statevector:
            return np.inf

        a = self._ret['estimation']
        # Note: Assuming that all iterations have the same number of shots
        shots = sum(self._ret['counts'][0].values())
        fisher_information = shots / (a * (1 - a)) * sum((2 * mk + 1)**2 for mk in self._evaluation_schedule)

        return fisher_information

    def _run(self):
        if self._quantum_instance.is_statevector:

            # run circuit on statevector simlator
            self.construct_circuits(measurement=False)
            ret = self._quantum_instance.execute(self._circuits)

            # get statevectors and construct MLE input
            state_vectors = [np.asarray(ret.get_statevector(circuit)) for circuit in self._circuits]
            self._ret['statevectors'] = state_vectors

            # evaluate results
            self._probabilities = self._evaluate_statevectors(state_vectors)
        else:
            # run circuit on QASM simulator
            self.construct_circuits(measurement=True)
            ret = self._quantum_instance.execute(self._circuits)

            # get counts and construct MLE input
            self._ret['counts'] = [ret.get_counts(circuit) for circuit in self._circuits]
            self._probabilities = self._evaluate_counts(self._ret['counts'])

        # run maximum likelihood estimation and construct results
        self._ret['theta'] = self._run_mle(self._probabilities)
        self._ret['estimation'] = np.sin(self._ret['theta'])**2
        self._ret['mapped_value'] = self.a_factory.value_to_estimation(self._ret['estimation'])
        self._ret['fisher_information'] = self._compute_fisher_information()

        alpha = 0.05
        normal_quantile = norm.ppf(1 - alpha / 2)
        confidence_interval = self._ret['estimation'] + normal_quantile / np.sqrt(self._ret['fisher_information']) * np.array([-1, 1])
        mapped_confidence_interval = [self.a_factory.value_to_estimation(bound) for bound in confidence_interval]
        self._ret['95%_confidence_interval'] = mapped_confidence_interval

        return self._ret
