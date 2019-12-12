
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
The Iterative Quantum Amplitude Estimation Algorithm.
"""


import logging
import numpy as np
from statsmodels.stats.proportion import proportion_confint

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.aqua import AquaError

from .ae_base import (AmplitudeEstimationBase)

logger = logging.getLogger(__name__)


class IterativeAmplitudeEstimation(AmplitudeEstimationBase):
    """
    The Iterative Quantum Amplitude Estimation Algorithm.
    """

    CONFIGURATION = {
        'name': 'IterativeAmplitudeEstimation',
        'description': 'Iterative Amplitude Estimation Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'IterativeAmplitudeEstimation_schema',
            'type': 'object',
            'properties': {
                'epsilon': {
                    'type': 'number',
                    'default': 0.01,
                    'minimum': 0.0
                },
                'alpha': {
                    'type': 'number',
                    'default': 0.05,
                    'minimum': 0.0
                },
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

    def __init__(self, epsilon, alpha, ci_method='beta', min_ratio=2, a_factory=None,
                 q_factory=None, i_objective=None):
        """
        Initializer.

        Args:
            epsilon (float): target precision for estimation target a
            alpha (float): confidence level
            ci_method (string): statistical method for confidence interval estimation
            min_ratio (float): minimal q-ratio (K_{i+1} / K_i) for FindNextK
            a_factory (CircuitFactory): A oracle
            q_factory (CircuitFactory): Q oracle
            i_objective (int): index of objective qubit
        """
        self.validate(locals())
        super().__init__(a_factory, q_factory, i_objective)

        # store parameters
        self._epsilon = epsilon
        self._alpha = alpha
        self._ci_method = ci_method
        self._min_ratio = min_ratio

    @property
    def precision(self):
        """
        Returns the target precision `epsilon` of the algorithm

        Returns:
            float: target precision
        """
        return self._epsilon

    @precision.setter
    def precision(self, epsilon):
        """
        Set the target precision of the algorithm.

        Args:
            epsilon (float): target precision for estimation target a
        """
        self._epsilon = epsilon

    def _find_next_k(self, k, up, theta_interval, min_ratio=2):
        """
        Find the largest integer k, such that the scaled interval (4k + 2)*theta_interval
        lies completely in [0, pi] or [pi, 2pi], for theta_interval = (theta_lower, theta_upper).

        Args:
            k (int): Current power of the Q operator.
            up (bool): Boolean flag of wheather theta lies in upper half-circle or not.
            theta_interval (tuple(float, float)): Current confidence interval for the angle
                theta, i.e. (theta_lower, theta_upper)
            min_ratio (float): Minimal ratio K/K_i allowed in the algorithm

        Returns:
            tuple(int, bool): Next power k, and boolean flag for extrapoled interval
        """
        if min_ratio <= 1:
            raise AquaError('min_ratio must be larger than 1: '
                            'the next k should not be smaller than the previous one')

        # intialize variables
        theta_l, theta_u = theta_interval
        K_current = 4 * k + 2

        # feasible K is not bigger than K_max, which is bounded by the length of current CI
        K_max = int(1 / (2 * (theta_u - theta_l)))
        K = K_max - (K_max - 2) % 4

        # find next feasible K = 4k+2
        while K >= min_ratio * K_current:
            theta_min = K * theta_l - int(K * theta_l)
            theta_max = K * theta_u - int(K * theta_u)
            if theta_max <= 1 / 2 and theta_min <= 1 / 2 and theta_max >= theta_min:
                # if extrapolated theta in upper half-circle
                up = True
                return ((K - 2) / 4, up)
            elif theta_max >= 1 / 2 and theta_min >= 1 / 2 and theta_max >= theta_min:
                # if extrapolated theta in lower half-circle
                up = False
                return ((K - 2) / 4, up)
            K = K - 4
        # if algorithm does not find new feasible k, return old k
        return (k, up)

    def construct_circuit(self, k, measurement=False):
        """
        Construct the circuit Q^k A |0>

        Args:
            k (int): the power of Q
            measurement (bool): boolean flag to indicate if measurements should be included in the
                circuits

        Returns:
            QuantumCircuit: the circuit Q^k A |0>
        """
        # set up circuit
        q = QuantumRegister(self.a_factory.num_target_qubits, 'q')
        circuit = QuantumCircuit(q, name='circuit')

        # get number of ancillas and add register if needed
        num_ancillas = np.maximum(self.a_factory.required_ancillas(),
                                  self.q_factory.required_ancillas())

        q_aux = None
        # pylint: disable=comparison-with-callable
        if num_ancillas > 0:
            q_aux = QuantumRegister(num_ancillas, 'aux')
            circuit.add_register(q_aux)

        # add classical register if needed
        if measurement:
            c = ClassicalRegister(1)
            circuit.add_register(c)

        # add A operator
        self.a_factory.build(circuit, q, q_aux)

        # add Q^k
        self.q_factory.build_power(circuit, q, k, q_aux)

        # add optional measurement
        if measurement:
            circuit.measure(q[self.i_objective], c[0])

        return circuit

    def _probability_to_measure_one(self, counts_or_statevector):
        """
        Convenient function to get the probability to measure '1' in the last qubit

        Args:
            counts_or_statevector (Union(dict, numpy.array, list)): either the counts dictionary
                returned from the qasm_simulator (with one measured qubit only!) or the statevector
                returned from the statevector_simulator

        Returns:
            Union(tuple(int, float), float): if a dict is given, it returns
                (#one-counts, #one-counts/#all-counts), otherwise Pr(measure '1' in the last qubit)
        """
        if isinstance(counts_or_statevector, dict):
            one_counts = counts_or_statevector.get('1', 0)
            return one_counts, one_counts / sum(counts_or_statevector.values())
        else:
            statevector = counts_or_statevector
            num_qubits = self.a_factory.num_target_qubits

            # sum over all amplitudes where the objective qubit is 1
            prob = 0
            for i, g in enumerate(statevector):
                if ('{:0%db}' % num_qubits).format(i)[-(1 + self.i_objective)] == '1':
                    prob = prob + np.abs(g)**2
            return prob

    def _chernoff_confint(self, a, shots, T, alpha):
        """
        Compute the Chernoff confidence interval for i.i.d. Bernoulli trials with `shots` samples:

            [a - eps, a + eps], where eps = sqrt(3 * log(2 * T / alpha) / shots)

        Args:
            a (float): the current estimate
            shots (int): the number of shots
            T (int): the maximum number of rounds, used to compute epsilon_a
            alpha (float): the confidence level, used to compute epsilon_a

        Returns:
            tuple(float, float): the Chernoff confidence interval
        """
        epsilon_a = np.sqrt(3 * np.log(2 * T / alpha) / shots)
        if a - epsilon_a < 0:
            a_min = 0
        else:
            a_min = a - epsilon_a
        if a + epsilon_a > 1:
            a_max = 1
        else:
            a_max = a + epsilon_a

        return a_min, a_max

    def _run(self):
        # check that A and Q operators are correctly set
        self.check_factories()

        # initialize memory variables
        powers = [0]  # list of powers k: Q^k, inum_iterationsialize with initial power: 0
        qs = []  # multiplication factors
        ups = [True]  # intially theta is in the upper half-circle
        theta_intervals = [[0, 1 / 4]]  # apriori knowledge of theta / 2 / pi
        a_intervals = [[0, 1]]  # apriori knowledge of a parameter
        num_oracle_queries = 0
        num_one_shots = []
        # maximum number of rounds
        T = int(np.log(self._min_ratio * np.pi / 8 / self._epsilon) / np.log(self._min_ratio)) + 1

        # for statevector we can directly return the probability to measure 1
        # note, that no iterations here are necessary
        if self._quantum_instance.is_statevector:
            # simulate circuit
            circuit = self.construct_circuit(k=0, measurement=False)
            ret = self._quantum_instance.execute(circuit)

            # get statevector
            statevector = ret.get_statevector(circuit)

            # calculate the probability of measuring '1'
            prob = self._probability_to_measure_one(statevector)

            a_confidence_interval = [prob, prob]
            a_intervals.append(a_confidence_interval)

            theta_i_interval = [np.arccos(1 - 2 * a_i) / 2 / np.pi for a_i in a_confidence_interval]
            theta_intervals.append(theta_i_interval)
            num_oracle_queries = 1

        else:
            num_iterations = 0  # keep track of the number of iterations
            shots = self._quantum_instance._run_config.shots  # number of shots per iteration

            # do while loop, keep in mind that we scaled theta mod 2pi such that it lies in [0,1]
            while theta_intervals[-1][1] - theta_intervals[-1][0] > self._epsilon / np.pi:
                num_iterations += 1

                # get the next k
                k, up = self._find_next_k(powers[-1], ups[-1], theta_intervals[-1],
                                          min_ratio=self._min_ratio)

                # store the variables
                powers.append(k)
                ups.append(up)
                qs.append((2 * powers[-1] + 1) / (2 * powers[-2] + 1))

                # run measurements for Q^k A|0> circuit
                circuit = self.construct_circuit(k, measurement=True)
                ret = self._quantum_instance.execute(circuit)

                # get the counts and store them
                counts = ret.get_counts(circuit)

                # calculate the probability of measuring '1', 'prob' is a_i in the paper
                one_counts, prob = self._probability_to_measure_one(counts)
                num_one_shots.append(one_counts)

                # track number of Q-oracle calls
                num_oracle_queries += shots * k

                # if on the previous iterations we have K_{i-1} == K_i, we sum these samples up
                j = 1  # number of times we stayed fixed at the same K
                round_shots = shots
                round_one_counts = one_counts
                if num_iterations > 1:
                    while powers[num_iterations - j] == powers[num_iterations] \
                            and num_iterations >= j + 1:
                        j = j + 1
                        round_shots += shots
                        round_one_counts += num_one_shots[-j]

                # compute a_min_i, a_max_i
                if self._ci_method == 'chernoff':
                    a_i_min, a_i_max = self._chernoff_confint(prob, round_shots, T, self._alpha)
                else:
                    a_i_min, a_i_max = proportion_confint(round_one_counts, round_shots,
                                                          method=self._ci_method,
                                                          alpha=self._alpha / T)

                # compute theta_min_i, theta_max_i
                if up:
                    theta_min_i = np.arccos(1 - 2 * a_i_min) / 2 / np.pi
                    theta_max_i = np.arccos(1 - 2 * a_i_max) / 2 / np.pi
                else:
                    theta_min_i = 1 - np.arccos(1 - 2 * a_i_max) / 2 / np.pi
                    theta_max_i = 1 - np.arccos(1 - 2 * a_i_min) / 2 / np.pi

                # compute theta_u, theta_l of this iteration
                K_i = 4 * k + 2  # current K_i factor
                theta_u = (int(K_i * theta_intervals[-1][1]) + theta_max_i) / K_i
                theta_l = (int(K_i * theta_intervals[-1][0]) + theta_min_i) / K_i
                theta_intervals.append([theta_l, theta_u])

                # compute a_u_i, a_l_i
                a_u = np.sin(2 * np.pi * theta_u)**2
                a_l = np.sin(2 * np.pi * theta_l)**2
                a_intervals.append([a_l, a_u])

        # get the latest confidence interval for the estimate of a
        a_confidence_interval = a_intervals[-1]

        # the final estimate is the mean of the confidence interval
        value = np.mean(a_confidence_interval)

        # transform to estimate
        estimation = self.a_factory.value_to_estimation(value)
        confidence_interval = [self.a_factory.value_to_estimation(x) for x in a_confidence_interval]

        # set up results dictionary
        results = {
            'value': value,
            'value_confidence_interval': a_confidence_interval,
            'confidence_interval': confidence_interval,
            'estimation': estimation,
            'alpha': self._alpha,
            'actual_epsilon': (confidence_interval[1] - confidence_interval[0]) / 2,
            'num_oracle_queries': num_oracle_queries,
            'a_intervals': a_intervals,
            'theta_intervals': theta_intervals,
            'powers': powers,
            'qs': qs,
        }

        return results
