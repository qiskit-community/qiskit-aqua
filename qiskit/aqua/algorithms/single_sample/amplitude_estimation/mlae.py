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
import numpy as np
from scipy.optimize import brute
from scipy.stats import norm, chi2

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.aqua import AquaError
from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class

from .ae_algorithm import AmplitudeEstimationAlgorithm

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class MaximumLikelihoodAmplitudeEstimation(AmplitudeEstimationAlgorithm):
    """
    The Amplitude Estimation without QPE algorithm.
    """

    CONFIGURATION = {
        'name': 'MaximumLikelihoodAmplitudeEstimation',
        'description': 'Maximum Likelihood Amplitude Estimation',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'MaximumLikelihoodAmplitudeEstimation_schema',
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

    def __init__(self, log_max_evals, a_factory=None, i_objective=None,
                 q_factory=None, likelihood_evals=None):
        """
        Constructor.

        Args:
            log_max_evals (int): base-2-logarithm of maximal number of evaluations -
                resulting evaluation schedule will be [Q^2^0, ..., Q^2^{max_evals_log-1}]
            a_factory (CircuitFactory): the CircuitFactory subclass object
                representing the problem unitary
            i_objective (int): index of qubit representing the objective in the uncertainty problem
            q_factory (CircuitFactory): the CircuitFactory subclass object representing
                an amplitude estimation sample (based on a_factory)
            likelihood_evals (int): The number of gridpoints for the maximum search
                of the likelihood function
        """
        self.validate(locals())
        super().__init__(a_factory, q_factory, i_objective)

        # get parameters
        self._log_max_evals = log_max_evals
        self._evaluation_schedule = [2**j for j in range(log_max_evals)]

        self._likelihood_evals = likelihood_evals
        # default number of evaluations is max(10^5, 10^5 * 2^(log_max_evals - 5))
        if likelihood_evals is None:
            default = 10000
            self._likelihood_evals = default * np.maximum(1, pow(2, log_max_evals - 5))

        self._circuits = []
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params (dict): parameters dictionary
            algo_input (AlgorithmInput): Input instance
        Returns:
            MaximumLikelihoodAmplitudeEstimation: instance of this class
        Raises:
            AquaError: input instance not supported
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

    @property
    def _num_qubits(self):
        if self.a_factory is None:  # if A factory is not set, no qubits are specified
            return 0

        num_ancillas = self.q_factory.required_ancillas_controlled()
        num_qubits = self.a_factory.num_target_qubits + num_ancillas

        return num_qubits

    def construct_circuits(self, measurement=False):
        """
        Construct the Amplitude Estimation w/o QPE quantum circuits.

        Args:
            measurement (bool): Boolean flag to indicate if measurement
                        should be included in the circuits.

        Returns:
            list: a list with the QuantumCircuit objects for the algorithm
        """
        # construct first part of circuit
        q = QuantumRegister(self.a_factory.num_target_qubits, 'q')

        qc_a = QuantumCircuit(q, name='qc_a')

        # get number of ancillas
        num_ancillas = np.maximum(self.a_factory.required_ancillas(),
                                  self.q_factory.required_ancillas())

        q_aux = None
        # pylint: disable=comparison-with-callable
        if num_ancillas > 0:
            q_aux = QuantumRegister(num_ancillas, 'aux')
            qc_a.add_register(q_aux)

        # add classical register if needed
        if measurement:
            c = ClassicalRegister(1)
            qc_a.add_register(c)

        self.a_factory.build(qc_a, q, q_aux)

        self._circuits = []
        for k in self._evaluation_schedule:
            qc_k = qc_a.copy(name='qc_a_q_%s' % k)
            self.q_factory.build_power(qc_k, q, k, q_aux)

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

    def _get_hits(self):
        """
        Get the good and total counts

        Returns:
            tuple(list, list): a pair of two lists,
            ([1-counts per experiment], [shots per experiment])
        Raises:
            AquaError: Call run() first
        """
        one_hits = []  # h_k: how often 1 has been measured, for a power Q^(m_k)
        all_hits = []  # N_k: how often has been measured at a power Q^(m_k)
        try:
            if self.quantum_instance.is_statevector:
                probabilities = self._evaluate_statevectors(self._ret['statevectors'])
                one_hits = probabilities
                all_hits = np.ones_like(one_hits)

            else:
                for c in self._ret['counts']:
                    one_hits += [c.get('1', 0)]  # return 0 if no key '1' found
                    all_hits += [sum(c.values())]
        except KeyError:
            raise AquaError('Call run() first!')

        return one_hits, all_hits

    def _safe_min(self, array, default=0):
        """
        Returns:
            float: default if array is empty, otherwise numpy.max(array)
        """
        if not array:
            return default
        return np.min(array)

    def _safe_max(self, array, default=(np.pi / 2)):
        """
        Returns:
            float: default if array is empty, otherwise numpy.max(array)
        """
        if not array:
            return default
        return np.max(array)

    def _compute_fisher_information(self, a=None, num_sum_terms=None, observed=False):
        """
        Compute the Fisher information.

        Args:
            a (float): a
            num_sum_terms (int): num sum terms
            observed (bool): If True, compute the observed Fisher information,
                otherwise the theoretical one

        Returns:
            float: The computed Fisher information, or np.inf if statevector
            simulation was used.
        Raises:
            KeyError: Call run() first!
        """
        # Set the value a. Use `est_a` if provided.
        if a is None:
            try:
                a = self._ret['value']
            except KeyError:
                raise KeyError("Call run() first!")

        # Corresponding angle to the value a (only use real part of 'a')
        theta_a = np.arcsin(np.sqrt(np.real(a)))

        # Get the number of hits (Nk) and one-hits (hk)
        one_hits, all_hits = self._get_hits()

        # Include all sum terms or just up to a certain term?
        evaluation_schedule = self._evaluation_schedule
        if num_sum_terms is not None:
            evaluation_schedule = evaluation_schedule[:num_sum_terms]
            # not necessary since zip goes as far as shortest list:
            # all_hits = all_hits[:num_sum_terms]
            # one_hits = one_hits[:num_sum_terms]

        # Compute the Fisher information
        fisher_information = None
        if observed:
            d_logL = 0
            for Nk, hk, mk in zip(all_hits, one_hits, evaluation_schedule):
                tan = np.tan((2 * mk + 1) * theta_a)
                d_logL += (2 * mk + 1) * (hk / tan + (Nk - hk) * tan)

            d_logL /= np.sqrt(a * (1 - a))
            fisher_information = d_logL**2 / len(all_hits)

        else:
            fisher_information = \
                1 / (a * (1 - a)) * sum(Nk * (2 * mk + 1)**2 for Nk, mk in zip(all_hits,
                                                                               evaluation_schedule))

        return fisher_information

    def _fisher_ci(self, alpha=0.05, observed=False):
        """
        Compute the alpha confidence interval based on the Fisher information

        Args:
            alpha (float): The level of the confidence interval (< 0.5)
            observed (bool): If True, use observed Fisher information

        Returns:
            float: The alpha confidence interval based on the Fisher information
        Raises:
            AssertionError: Call run() first!
        """
        # Get the (observed) Fisher information
        fisher_information = None
        try:
            fisher_information = self._ret['fisher_information']
        except KeyError:
            raise AssertionError("Call run() first!")

        if observed:
            fisher_information = self._compute_fisher_information(observed=True)

        normal_quantile = norm.ppf(1 - alpha / 2)
        ci = np.real(self._ret['value']) + \
            normal_quantile / np.sqrt(fisher_information) * np.array([-1, 1])
        mapped_ci = [self.a_factory.value_to_estimation(bound) for bound in ci]
        return mapped_ci

    def _likelihood_ratio_ci(self, alpha=0.05, nevals=10000):
        """
        Compute the likelihood-ratio confidence interval.

        Args:
            alpha (float): the level of the confidence interval (< 0.5)
            nevals (int): the number of evaluations to find the
                intersection with the loglikelihood function

        Returns:
            float: The alpha-likelihood-ratio confidence interval.
        """

        def loglikelihood(theta, one_counts, all_counts):
            logL = 0
            for i, k in enumerate(self._evaluation_schedule):
                logL += np.log(np.sin((2 * k + 1) * theta) ** 2) * one_counts[i]
                logL += np.log(np.cos((2 * k + 1) * theta) ** 2) * (all_counts[i] - one_counts[i])
            return logL

        one_counts, all_counts = self._get_hits()

        thetas = np.linspace(np.pi / nevals / 2, np.pi / 2, nevals)
        values = np.zeros(len(thetas))
        for i, t in enumerate(thetas):
            values[i] = loglikelihood(t, one_counts, all_counts)

        loglik_mle = loglikelihood(self._ret['theta'], one_counts, all_counts)
        chi2_quantile = chi2.ppf(1 - alpha, df=1)
        thres = loglik_mle - chi2_quantile / 2

        # the (outer) LR confidence interval
        above_thres = thetas[values >= thres]

        # it might happen that the `above_thres` array is empty,
        # to still provide a valid result use safe_min/max which
        # then yield [0, pi/2]
        ci = [self._safe_min(above_thres, default=0),
              self._safe_max(above_thres, default=(np.pi / 2))]
        mapped_ci = [self.a_factory.value_to_estimation(np.sin(bound)**2) for bound in ci]

        return mapped_ci

    def confidence_interval(self, alpha, kind='fisher'):
        """
        Proxy calling the correct method to compute the confidence interval,
        according to the value of `kind`
        """
        # check if AE did run already
        if 'estimation' not in self._ret.keys():
            raise AquaError('Call run() first!')

        # if statevector simulator the estimate is exact
        if self._quantum_instance.is_statevector:
            return 2 * [self._ret['estimation']]

        if kind in ['likelihood_ratio', 'lr']:
            return self._likelihood_ratio_ci(alpha)

        if kind in ['fisher', 'fi']:
            return self._fisher_ci(alpha, observed=False)

        if kind in ['observed_fisher', 'observed_information', 'oi']:
            return self._fisher_ci(alpha, observed=True)

        raise NotImplementedError('CI `{}` is not implemented.'.format(kind))

    def _compute_mle_safe(self):
        """
        Compute the MLE via a grid-search. This is a stable approach if
        sufficient gridpoints are used (usually > 10'000).
        """
        one_hits, all_hits = self._get_hits()

        # search range
        eps = 1e-15  # to avoid division by 0
        search_range = [0 + eps, np.pi / 2 - eps]

        def loglikelihood(theta):
            # logL contains the first `it` terms of the full loglikelihood
            logL = 0
            for i, k in enumerate(self._evaluation_schedule):
                logL += np.log(np.sin((2 * k + 1) * theta) ** 2) * one_hits[i]
                logL += np.log(np.cos((2 * k + 1) * theta) ** 2) * (all_hits[i] - one_hits[i])
            return -logL

        est_theta = brute(loglikelihood, [search_range], Ns=self._likelihood_evals)[0]
        return est_theta

    def _run_mle(self):
        # TODO implement a **reliable**, fast method to find the maximum of the likelihood function
        return self._compute_mle_safe()

    def _run(self):
        # check if A factory has been set
        if self.a_factory is None:
            raise AquaError("a_factory must be set!")

        if self._quantum_instance.is_statevector:

            # run circuit on statevector simulator
            self.construct_circuits(measurement=False)
            ret = self._quantum_instance.execute(self._circuits)

            # get statevectors and construct MLE input
            state_vectors = [np.asarray(ret.get_statevector(circuit)) for circuit in self._circuits]
            self._ret['statevectors'] = state_vectors

        else:
            # run circuit on QASM simulator
            self.construct_circuits(measurement=True)
            ret = self._quantum_instance.execute(self._circuits)

            # get counts and construct MLE input
            self._ret['counts'] = [ret.get_counts(circuit) for circuit in self._circuits]

        # run maximum likelihood estimation and construct results
        self._ret['theta'] = self._run_mle()
        self._ret['value'] = np.sin(self._ret['theta'])**2
        self._ret['estimation'] = self.a_factory.value_to_estimation(self._ret['value'])
        self._ret['fisher_information'] = self._compute_fisher_information()

        confidence_interval = self._fisher_ci(alpha=0.05)
        self._ret['95%_confidence_interval'] = confidence_interval

        return self._ret
