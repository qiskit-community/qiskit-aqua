
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
from scipy.stats import beta

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.aqua import AquaError

from .ae_algorithm import AmplitudeEstimationAlgorithm

logger = logging.getLogger(__name__)


class IterativeAmplitudeEstimation(AmplitudeEstimationAlgorithm):
    """
    This class implements the Iterative Quantum Amplitude Estimation (QAE) algorithm, proposed
    in https://arxiv.org/abs/1912.05559. The output of the algorithm is an estimate that,
    with at least probability 1 - alpha, differs by epsilon to the target value, where
    both alpha and epsilon can be specified.

    It differs from the original QAE algorithm proposed by Brassard
    (https://arxiv.org/abs/quant-ph/0005055) in that it does not rely on Quantum Phase Estimation,
    but is only based on Grover's algorithm. Iterative IQAE iteratively applies carefully selected
    Grover iterations to find an estimate for the target amplitude.
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
                    'minimum': 0.0,
                    'maximum': 0.5,
                },
                'alpha': {
                    'type': 'number',
                    'default': 0.05,
                    'minimum': 0.0,
                    'maximum': 1.0,
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

        The output of the algorithm is an estimate for the amplitude `a`, that with at least
        probability 1 - alpha has an error of epsilon. The number of A operator calls scales
        linearly in 1/epsilon (up to a logarithmic factor).

        Args:
            epsilon (float): target precision for estimation target `a`
            alpha (float): confidence level, the target probability is 1 - alpha
            ci_method (str): statistical method used to estimate the confidence intervals in each
                iteration, can be 'chernoff' for the Chernoff intervals or 'beta' for the
                Clopper-Pearson intervals (default)
            min_ratio (float): minimal q-ratio (K_{i+1} / K_i) for FindNextK
            a_factory (CircuitFactory): the A operator, specifying the QAE problem
            q_factory (CircuitFactory): the Q operator (Grover operator), constructed from the
                A operator
            i_objective (int): index of the objective qubit, that marks the 'good/bad' states

        Raises:
            AquaError: if the method to compute the confidence intervals is not supported
        """
        self.validate(locals())
        super().__init__(a_factory, q_factory, i_objective)

        # store parameters
        self._epsilon = epsilon
        self._alpha = alpha
        self._min_ratio = min_ratio

        if ci_method in ['chernoff', 'beta']:
            self._ci_method = ci_method
        else:
            raise AquaError('invalid method to compute confidence intervals')

        # results dictionary
        self._ret = {}

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

    def _find_next_k(self, k, upper_half_circle, theta_interval, min_ratio=2):
        """
        Find the largest integer k, such that the scaled interval (4k + 2)*theta_interval
        lies completely in [0, pi] or [pi, 2pi], for theta_interval = (theta_lower, theta_upper).

        Args:
            k (int): current power of the Q operator
            upper_half_circle (bool): boolean flag of whether theta lies in upper half-circle or not
            theta_interval (tuple(float, float)): current confidence interval for the angle
                theta, i.e. (theta_lower, theta_upper)
            min_ratio (float): minimal ratio K/K_i allowed in the algorithm

        Returns:
            tuple(int, bool): next power k, and boolean flag for the extrapolated interval

        Raises:
            AquaError: if min_ratio is smaller or equal to 1
        """
        if min_ratio <= 1:
            raise AquaError('min_ratio must be larger than 1: '
                            'the next k should not be smaller than the previous one')

        # initialize variables
        theta_l, theta_u = theta_interval
        old_scaling = 4 * k + 2  # current K_i factor

        # the largest feasible K cannot be larger than K_max, which is bounded by the length of
        # the current confidence interval
        max_scaling = int(1 / (2 * (theta_u - theta_l)))
        scaling = max_scaling - (max_scaling - 2) % 4

        # find next feasible K = 4k+2
        while scaling >= min_ratio * old_scaling:
            theta_min = scaling * theta_l - int(scaling * theta_l)
            theta_max = scaling * theta_u - int(scaling * theta_u)

            if theta_min <= theta_max <= 0.5 and theta_min <= 0.5:
                # if extrapolated theta is in upper half-circle
                upper_half_circle = True
                return int((scaling - 2) / 4), upper_half_circle

            elif theta_max >= 0.5 and theta_max >= theta_min >= 0.5:
                # if extrapolated theta is in lower half-circle
                upper_half_circle = False
                return int((scaling - 2) / 4), upper_half_circle

            scaling -= 4

        # if we do not find a feasible k, return the old one
        return int(k), upper_half_circle

    def construct_circuit(self, k, measurement=False):
        """
        Construct the circuit Q^k A |0>, with the A operator specifying the QAE problem and
        the Grover operator Q.

        Args:
            k (int): the power of Q operator
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
            return int(one_counts), one_counts / sum(counts_or_statevector.values())
        else:
            statevector = counts_or_statevector
            num_qubits = self.a_factory.num_target_qubits

            # sum over all amplitudes where the objective qubit is 1
            prob = 0
            for i, amplitude in enumerate(statevector):
                if ('{:0%db}' % num_qubits).format(i)[-(1 + self.i_objective)] == '1':
                    prob = prob + np.abs(amplitude)**2

            return prob

    def _chernoff_confint(self, value, shots, max_rounds, alpha):
        """
        Compute the Chernoff confidence interval for i.i.d. Bernoulli trials with `shots` samples:

            [value - eps, value + eps], where eps = sqrt(3 * log(2 * max_rounds/ alpha) / shots)

        but at most [0, 1].

        Args:
            value (float): the current estimate
            shots (int): the number of shots
            max_rounds (int): the maximum number of rounds, used to compute epsilon_a
            alpha (float): the confidence level, used to compute epsilon_a

        Returns:
            tuple(float, float): the Chernoff confidence interval
        """
        eps = np.sqrt(3 * np.log(2 * max_rounds / alpha) / shots)
        lower = np.maximum(0, value - eps)
        upper = np.minimum(1, value + eps)
        return lower, upper

    def _clopper_pearson_confint(self, counts, shots, alpha):
        lower, upper = 0, 1

        # if counts == 0, the beta quantile returns nan
        if counts != 0:
            lower = beta.ppf(alpha / 2, counts, shots - counts + 1)

        # if counts == shots, the beta quantile returns nan
        if counts != shots:
            upper = beta.ppf(1 - alpha / 2, counts + 1, shots - counts)

        return lower, upper

    def _run(self):
        # check if A factory has been set
        if self.a_factory is None:
            raise AquaError("a_factory must be set!")

        # initialize memory variables
        powers = [0]  # list of powers k: Q^k, (called 'k' in paper)
        ratios = []  # list of multiplication factors (called 'q' in paper)
        theta_intervals = [[0, 1 / 4]]  # a priori knowledge of theta / 2 / pi
        a_intervals = [[0, 1]]  # a priori knowledge of the confidence interval of the estimate a
        num_oracle_queries = 0
        num_one_shots = []

        # maximum number of rounds
        max_rounds = int(np.log(self._min_ratio * np.pi / 8
                                / self._epsilon) / np.log(self._min_ratio)) + 1
        upper_half_circle = True  # initially theta is in the upper half-circle

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
                k, upper_half_circle = self._find_next_k(powers[-1], upper_half_circle,
                                                         theta_intervals[-1],
                                                         min_ratio=self._min_ratio)

                # store the variables
                powers.append(k)
                ratios.append((2 * powers[-1] + 1) / (2 * powers[-2] + 1))

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
                    a_i_min, a_i_max = self._chernoff_confint(prob, round_shots, max_rounds,
                                                              self._alpha)
                else:  # 'beta'
                    a_i_min, a_i_max = self._clopper_pearson_confint(round_one_counts, round_shots,
                                                                     self._alpha / max_rounds)

                # compute theta_min_i, theta_max_i
                if upper_half_circle:
                    theta_min_i = np.arccos(1 - 2 * a_i_min) / 2 / np.pi
                    theta_max_i = np.arccos(1 - 2 * a_i_max) / 2 / np.pi
                else:
                    theta_min_i = 1 - np.arccos(1 - 2 * a_i_max) / 2 / np.pi
                    theta_max_i = 1 - np.arccos(1 - 2 * a_i_min) / 2 / np.pi

                # compute theta_u, theta_l of this iteration
                scaling = 4 * k + 2  # current K_i factor
                theta_u = (int(scaling * theta_intervals[-1][1]) + theta_max_i) / scaling
                theta_l = (int(scaling * theta_intervals[-1][0]) + theta_min_i) / scaling
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

        # add result items to the results dictionary
        self._ret = {
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
            'ratios': ratios,
        }

        return self._ret
