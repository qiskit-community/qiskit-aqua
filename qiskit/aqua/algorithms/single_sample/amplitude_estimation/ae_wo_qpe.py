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
from scipy.optimize import minimize
from scipy.stats import norm, chi2

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.aqua import AquaError
from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class

from .ae_base import AmplitudeEstimationBase

logger = logging.getLogger(__name__)


class AmplitudeEstimationWithoutQPE(AmplitudeEstimationBase):
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
        super().__init__(a_factory, q_factory, i_objective)

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

    def _get_hits(self, counts):

        one_hits = []  # h_k: how often 1 has been measured, for a power Q^(m_k)
        all_hits = []  # N_k: how often has been measured at a power Q^(m_k)
        for c in counts:
            one_hits += [c.get('1', 0)]  # return 0 if no key '1' found
            all_hits += [sum(c.values())]

        return one_hits, all_hits

    def _run_mle(self):
        """
        Proxy to call the suitable MLE for statevector or qasm simulator.
        """
        if self._quantum_instance.is_statevector:
            return self._run_mle_statevector()

        return self._run_mle_counts()

    def _run_mle_statevector(self):
        """
        Find the MLE if statevector simulation is used.
        Instead of shrinking the interval using the Fisher information,
        which we cannot do here, use the theta estimate of the previous
        iteration as the initial guess of the next one.
        With several iterations this should converge reliably to the maximum.

        Returns:
            MLE for a statevector simulation
        """
        probs = self._evaluate_statevectors(self._ret['statevectors'])

        search_range = [0, np.pi / 2]
        init = np.mean(search_range)
        best_theta = None

        for it in range(len(self._evaluation_schedule)):
            def loglikelihood(theta):
                logL = 0
                for i, k in enumerate(self._evaluation_schedule[:it + 1]):
                    logL = np.log(np.sin((2 * k + 1) * theta) ** 2) * probs[i] \
                        + np.log(np.cos((2 * k + 1) * theta) ** 2) * (1 - probs[i])
                return -logL

            # find the current optimum, this is our new initial point
            res = minimize(loglikelihood, init, bounds=[search_range], method="SLSQP")
            init = res.x

            # keep track of the best theta estimate
            if best_theta is None:
                best_theta = res.x
            elif res.fun < loglikelihood(best_theta):
                best_theta = res.x

        return best_theta[0]  # return the value, not a 1d numpy.array

    def _run_mle_counts(self):
        """
        Compute the MLE for a shot-based simulation.

        Returns:
            The MLE for a shot-based simulation.
        """
        # the number of times 1 has been measured and the total number
        # of measurements
        one_hits, all_hits = self._get_hits(self._ret['counts'])

        # empirical factor of how large the search range will be
        confidence_level = 5

        # initial search range
        eps = 1e-15  # to avoid division by 0
        search_range = [0 + eps, np.pi / 2 - eps]

        est_theta = None

        for it in range(len(self._evaluation_schedule)):
            def loglikelihood(theta):
                # logL contains the first `it` terms of the full loglikelihood
                logL = 0
                for i, k in enumerate(self._evaluation_schedule[:it + 1]):
                    logL += np.log(np.sin((2 * k + 1) * theta) ** 2) * one_hits[i]
                    logL += np.log(np.cos((2 * k + 1) * theta) ** 2) * (all_hits[i] - one_hits[i])
                return -logL

            # crudely find the optimum
            est_theta = minimize(loglikelihood, np.mean(search_range), bounds=[search_range]).x
            est_a = np.sin(est_theta)**2

            # estimate the error of the est_theta
            fisher_information = self._compute_fisher_information(est_a, it + 1)
            est_error_a = 1 / np.sqrt(fisher_information)
            est_error_theta = est_error_a / (2 * np.sqrt(est_error_a) * np.sqrt(1 - est_error_a**2))

            # update the range
            search_range[0] = np.maximum(0 + eps, est_theta - confidence_level * est_error_theta)
            search_range[1] = np.minimum(np.pi / 2 - eps, est_theta + confidence_level * est_error_theta)

        return est_theta

    def _save_min(self, array, default=0):
        if len(array) == 0:
            return default
        return np.min(array)

    def _save_max(self, array, default=(np.pi / 2)):
        if len(array) == 0:
            return default
        return np.max(array)

    def _likelihood_ratio_ci(self, alpha=0.05, nevals=10000):
        """
        Compute the likelihood-ratio confidence interval.

        Args:
            alpha (float): the level of the confidence interval (< 0.5)
            nevals (int): the number of evaluations to find the
                intersection with the loglikelihood function

        Returns:
            The alpha-likelihood-ratio confidence interval.
        """

        def loglikelihood(theta, one_counts, all_counts):
            logL = 0
            for i, k in enumerate(self._evaluation_schedule):
                logL += np.log(np.sin((2 * k + 1) * theta) ** 2) * one_counts[i]
                logL += np.log(np.cos((2 * k + 1) * theta) ** 2) * (all_counts[i] - one_counts[i])
            return logL

        one_counts, all_counts = self._get_hits(self._ret['counts'])

        thetas = np.linspace(np.pi / nevals / 2, np.pi / 2, nevals)
        values = np.zeros(len(thetas))
        for i, t in enumerate(thetas):
            values[i] = loglikelihood(t, one_counts, all_counts)

        loglik_mle = loglikelihood(self._ret['theta'], one_counts, all_counts)
        chi2_quantile = chi2.ppf(1 - alpha, df=1)
        thres = loglik_mle - chi2_quantile / 2

        # the outer LR confidence interval
        above_thres = thetas[values >= thres]

        # it might happen that the `above_thres` array is empty,
        # to still provide a valid result use save_min/max which
        # then yield [0, pi/2]
        ci_outer = [self._save_min(above_thres, default=0),
                    self._save_max(above_thres, default=(np.pi / 2))]
        mapped_ci_outer = [self.a_factory.value_to_estimation(np.sin(bound)**2) for bound in ci_outer]

        # the inner LR confidence interval:
        # [largest value below mle and above thres, smallest value above mle and above thres]
        larger_than_mle = above_thres[above_thres > self._ret['theta']]
        smaller_than_mle = above_thres[above_thres < self._ret['theta']]
        ci_inner = [self._save_max(smaller_than_mle, default=0),
                    self._save_min(larger_than_mle, default=(np.pi / 2))]
        mapped_ci_inner = [self.a_factory.value_to_estimation(np.sin(bound)**2) for bound in ci_inner]

        return mapped_ci_outer, mapped_ci_inner

    def _fisher_ci(self, alpha=0.05, observed=False):
        """
        Compute the alpha confidence interval based on the Fisher information

        Args:
            alpha (float): The level of the confidence interval (< 0.5)
            observed (bool): If True, use observed Fisher information

        Returns:
            The alpha confidence interval based on the Fisher information
        """
        # Get the (observed) Fisher information
        fisher_information = None
        try:
            fisher_information = self._ret["fisher_information"]
        except KeyError:
            raise AssertionError("Call run() first!")

        if observed:
            fisher_information = self._compute_fisher_information(observed=True)

        normal_quantile = norm.ppf(1 - alpha / 2)
        ci = self._ret['estimation'] + normal_quantile / np.sqrt(fisher_information) * np.array([-1, 1])
        mapped_ci = [self.a_factory.value_to_estimation(bound) for bound in ci]
        return mapped_ci

    def _compute_fisher_information(self, a=None, num_sum_terms=None, observed=False):
        """
        Compute the Fisher information.

        Args:
            observed (bool): If True, compute the observed Fisher information,
                otherwise the theoretical one

        Returns:
            The computed Fisher information, or np.inf if statevector
            simulation was used.
        """
        # the fisher information is infinite, since:
        # 1) statevector simulation should return the exact value
        # 2) statevector probabilities correspond to "infinite" shots
        if self._quantum_instance.is_statevector:
            return np.inf

        # Set the value a. Use `est_a` if provided.
        if a is None:
            try:
                a = self._ret['estimation']
            except KeyError:
                raise KeyError("Call run() first!")

        # Corresponding angle to the value a.
        theta_a = np.arcsin(np.sqrt(a))

        # Get the number of hits (Nk) and one-hits (hk)
        one_hits, all_hits = self._get_hits(self._ret['counts'])

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
            fisher_information = 1 / (a * (1 - a)) * sum(Nk * (2 * mk + 1)**2 for Nk, mk in zip(all_hits, evaluation_schedule))

        return fisher_information

    def confidence_interval(self, alpha, kind='fisher'):
        # check if AE did run already
        if 'mle' not in self._ret.keys():
            raise AquaError('Call run() first!')

        if kind in ['likelihood_ratio', 'lr']:
            return self._likelihood_ratio_ci(alpha)

        if kind in ['fisher', 'fi']:
            return self._fisher_ci(alpha, observed=False)

        if kind in ['observed_fisher', 'observed_information', 'oi']:
            return self._fisher_ci(alpha, observed=True)

        raise NotImplementedError('CI `{}` is not implemented.'.format(kind))

    def _run(self):
        self.check_factories()

        if self._quantum_instance.is_statevector:

            # run circuit on statevector simlator
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
        self._ret['estimation'] = np.sin(self._ret['theta'])**2
        self._ret['mapped_value'] = self.a_factory.value_to_estimation(self._ret['estimation'])
        self._ret['fisher_information'] = self._compute_fisher_information()

        confidence_interval = self.compute_fisher_ci(alpha=0.05)
        self._ret['95%_confidence_interval'] = confidence_interval

        return self._ret
