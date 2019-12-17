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
from scipy.stats import chi2, norm
from scipy.optimize import bisect

from qiskit.aqua import AquaError
from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.circuits import PhaseEstimationCircuit
from qiskit.aqua.components.iqfts import Standard

from .ae_algorithm import AmplitudeEstimationAlgorithm
from .ae_utils import pdf_a, derivative_log_pdf_a, bisect_max

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class AmplitudeEstimation(AmplitudeEstimationAlgorithm):
    """
    The Amplitude Estimation algorithm.
    """

    CONFIGURATION = {
        'name': 'AmplitudeEstimation',
        'description': 'Amplitude Estimation Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
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
            {
                'pluggable_type': 'iqft',
                'default': {
                    'name': 'STANDARD',
                }
            },
        ],
    }

    def __init__(self, num_eval_qubits, a_factory=None,
                 i_objective=None, q_factory=None, iqft=None):
        """
        Initializer.

        Args:
            num_eval_qubits (int): number of evaluation qubits
            a_factory (CircuitFactory): the CircuitFactory subclass object representing
                                        the problem unitary
            i_objective (int): i objective
            q_factory (CircuitFactory): the CircuitFactory subclass object representing an
                                        amplitude estimation sample (based on a_factory)
            iqft (IQFT): the Inverse Quantum Fourier Transform pluggable component,
                            defaults to using a standard iqft when None
        """
        self.validate(locals())
        super().__init__(a_factory, q_factory, i_objective)

        # get parameters
        self._m = num_eval_qubits
        self._M = 2 ** num_eval_qubits

        if iqft is None:
            iqft = Standard(self._m)

        self._iqft = iqft
        self._circuit = None
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params (dict): parameters dictionary
            algo_input (AlgorithmInput): Input instance
        Returns:
            AmplitudeEstimation: instance of this class
        Raises:
            AquaError: Input instance not supported
        """
        if algo_input is not None:
            raise AquaError('Input instance not supported.')

        ae_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        num_eval_qubits = ae_params.get('num_eval_qubits')

        # Set up uncertainty problem. The params can include an uncertainty model
        # type dependent on the uncertainty problem and is this its responsibility
        # to create for itself from the complete params set that is passed to it.
        uncertainty_problem_params = params.get(
            Pluggable.SECTION_KEY_UNCERTAINTY_PROBLEM)
        uncertainty_problem = get_pluggable_class(
            PluggableType.UNCERTAINTY_PROBLEM,
            uncertainty_problem_params['name']).init_params(params)

        # Set up iqft, we need to add num qubits to params which is our num_ancillae bits here
        iqft_params = params.get(Pluggable.SECTION_KEY_IQFT)
        iqft_params['num_qubits'] = num_eval_qubits
        iqft = get_pluggable_class(
            PluggableType.IQFT, iqft_params['name']).init_params(params)

        return cls(num_eval_qubits, uncertainty_problem, q_factory=None, iqft=iqft)

    @property
    def _num_qubits(self):
        if self.a_factory is None:  # if A factory is not set, no qubits are specified
            return 0

        num_ancillas = self.q_factory.required_ancillas_controlled()
        num_qubits = self.a_factory.num_target_qubits + self._m + num_ancillas

        return num_qubits

    def construct_circuit(self, measurement=False):
        """
        Construct the Amplitude Estimation quantum circuit.

        Args:
            measurement (bool): Boolean flag to indicate if measurement
                should be included in the circuit.

        Returns:
            QuantumCircuit: the QuantumCircuit object for the constructed circuit
        """
        pec = PhaseEstimationCircuit(
            iqft=self._iqft, num_ancillae=self._m,
            state_in_circuit_factory=self.a_factory,
            unitary_circuit_factory=self.q_factory
        )

        self._circuit = pec.construct_circuit(measurement=measurement)
        return self._circuit

    def _evaluate_statevector_results(self, probabilities):
        # map measured results to estimates
        y_probabilities = OrderedDict()
        for i, probability in enumerate(probabilities):
            b = '{0:b}'.format(i).rjust(self._num_qubits, '0')[::-1]
            y = int(b[:self._m], 2)
            y_probabilities[y] = y_probabilities.get(y, 0) + probability

        a_probabilities = OrderedDict()
        for y, probability in y_probabilities.items():
            if y >= int(self._M / 2):
                y = self._M - y
            a = np.round(np.power(np.sin(y * np.pi / 2 ** self._m), 2),
                         decimals=7)
            a_probabilities[a] = a_probabilities.get(a, 0) + probability

        return a_probabilities, y_probabilities

    def _compute_fisher_information(self, observed=False):
        fisher_information = None
        mlv = self._ret['ml_value']  # MLE in [0,1]
        m = self._m
        if observed:
            ai = np.asarray(self._ret['values'])
            pi = np.asarray(self._ret['probabilities'])

            # Calculate the observed Fisher information
            fisher_information = sum(p * derivative_log_pdf_a(a, mlv, m)**2 for p, a in zip(pi, ai))
        else:
            def integrand(x):
                return (derivative_log_pdf_a(x, mlv, m))**2 * pdf_a(x, mlv, m)

            M = 2**m
            grid = np.sin(np.pi * np.arange(M / 2 + 1) / M)**2
            fisher_information = sum(integrand(x) for x in grid)

        return fisher_information

    def _fisher_ci(self, alpha, observed=False):
        shots = self._ret['shots']
        mle = self._ret['ml_value']

        std = np.sqrt(shots * self._compute_fisher_information(observed))
        ci = mle + norm.ppf(1 - alpha / 2) / std * np.array([-1, 1])

        return [self.a_factory.value_to_estimation(bound) for bound in ci]

    def _likelihood_ratio_ci(self, alpha):
        # Compute the two intervals in which we the look for values above
        # the likelihood ratio: the two bubbles next to the QAE estimate
        M = 2**self._m
        qae = self._ret['value']
        y = M * np.arcsin(np.sqrt(qae)) / np.pi
        left_of_qae = np.sin(np.pi * (y - 1) / M)**2
        right_of_qae = np.sin(np.pi * (y + 1) / M)**2

        bubbles = [left_of_qae, qae, right_of_qae]

        # likelihood function
        ai = np.asarray(self._ret['values'])
        pi = np.asarray(self._ret['probabilities'])
        m = self._m
        shots = self._ret['shots']

        def loglikelihood(a):
            return np.sum(shots * pi * np.log(pdf_a(ai, a, m)))

        # The threshold above which the likelihoods are in the
        # confidence interval
        loglik_mle = loglikelihood(self._ret['ml_value'])
        thres = loglik_mle - chi2.ppf(1 - alpha, df=1) / 2

        def cut(x):
            return loglikelihood(x) - thres

        # Store the boundaries of the confidence interval
        lower = upper = self._ret['ml_value']

        # Check the two intervals/bubbles: check if they surpass the
        # threshold and if yes add the part that does to the CI
        for a, b in zip(bubbles[:-1], bubbles[1:]):
            # Compute local maximum and perform a bisect search between
            # the local maximum and the bubble boundaries
            locmax, val = bisect_max(loglikelihood, a, b, retval=True)
            if val >= thres:
                # Bisect pre-condition is that the function has different
                # signs at the boundaries of the interval we search in
                if cut(a) * cut(locmax) < 0:
                    left = bisect(cut, a, locmax)
                    lower = np.minimum(lower, left)
                if cut(locmax) * cut(b) < 0:
                    right = bisect(cut, locmax, b)
                    upper = np.maximum(upper, right)

        # Put together CI
        ci = [lower, upper]
        return [self.a_factory.value_to_estimation(bound) for bound in ci]

    def confidence_interval(self, alpha, kind='likelihood_ratio'):
        """
        Compute the (1 - alpha) confidence interval

        Args:
            alpha (float): confidence level: compute the (1 - alpha) confidence interval
            kind (str): the method to compute the confidence interval, can be 'fisher',
                'observed_fisher' or 'likelihood_ratio' (default)

        Returns:
            list[float]: the (1 - alpha) confidence interval

        Raises:
            AquaError: if 'mle' is not in self._ret.keys() (i.e. `run` was not called yet)
            NotImplementedError: if the confidence interval method `kind` is not implemented
        """
        # check if AE did run already
        if 'mle' not in self._ret.keys():
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

    def _run_mle(self):
        """
        Compute the Maximum Likelihood Estimator (MLE)

        Returns:
            The MLE for the previous AE run

        Note: Before calling this method, call the method `run` of the AmplitudeEstimation instance
        """
        M = self._M
        qae = self._ret['value']

        # likelihood function
        ai = np.asarray(self._ret['values'])
        pi = np.asarray(self._ret['probabilities'])
        m = self._m
        shots = self._ret['shots']

        def loglikelihood(a):
            return np.sum(shots * pi * np.log(pdf_a(ai, a, m)))

        # y is pretty much an integer, but to map 1.9999 to 2 we must first
        # use round and then int conversion
        y = int(np.round(M * np.arcsin(np.sqrt(qae)) / np.pi))

        # Compute the two intervals in which are candidates for containing
        # the maximum of the log-likelihood function: the two bubbles next to
        # the QAE estimate
        bubbles = None
        if y == 0:
            right_of_qae = np.sin(np.pi * (y + 1) / M)**2
            bubbles = [qae, right_of_qae]

        elif y == int(M / 2):
            left_of_qae = np.sin(np.pi * (y - 1) / M)**2
            bubbles = [left_of_qae, qae]

        else:
            left_of_qae = np.sin(np.pi * (y - 1) / M)**2
            right_of_qae = np.sin(np.pi * (y + 1) / M)**2
            bubbles = [left_of_qae, qae, right_of_qae]

        # Find global maximum amongst the two local maxima
        a_opt = qae
        loglik_opt = loglikelihood(a_opt)
        for a, b in zip(bubbles[:-1], bubbles[1:]):
            locmax, val = bisect_max(loglikelihood, a, b, retval=True)
            if val > loglik_opt:
                a_opt = locmax
                loglik_opt = val

        # Convert the value to an estimation
        val_opt = self.a_factory.value_to_estimation(a_opt)

        # Store MLE and the MLE mapped to an estimation
        self._ret['ml_value'] = a_opt
        self._ret['mle'] = val_opt

    def _run(self):
        # check if A factory has been set
        if self.a_factory is None:
            raise AquaError("a_factory must be set!")

        if self._quantum_instance.is_statevector:
            self.construct_circuit(measurement=False)
            # run circuit on statevector simulator
            ret = self._quantum_instance.execute(self._circuit)
            state_vector = np.asarray([ret.get_statevector(self._circuit)])
            self._ret['statevector'] = state_vector

            # get state probabilities
            state_probabilities = np.real(
                state_vector.conj() * state_vector)[0]

            # evaluate results
            a_probabilities, y_probabilities = self._evaluate_statevector_results(
                state_probabilities)

            # store number of shots: convention is 1 shot for statevector,
            # needed so that MLE works!
            self._ret['shots'] = 1
        else:
            # run circuit on QASM simulator
            self.construct_circuit(measurement=True)
            ret = self._quantum_instance.execute(self._circuit)

            # get counts
            self._ret['counts'] = ret.get_counts()

            # construct probabilities
            y_probabilities = {}
            a_probabilities = {}
            shots = sum(ret.get_counts().values())
            for state, counts in ret.get_counts().items():
                y = int(state.replace(' ', '')[:self._m][::-1], 2)
                p = counts / shots
                y_probabilities[y] = p
                a = np.round(np.power(np.sin(y * np.pi / 2 ** self._m), 2),
                             decimals=7)
                a_probabilities[a] = a_probabilities.get(a, 0.0) + p

            # store shots
            self._ret['shots'] = shots

        # construct a_items and y_items
        a_items = [(a, p) for (a, p) in a_probabilities.items() if p > 1e-6]
        y_items = [(y, p) for (y, p) in y_probabilities.items() if p > 1e-6]
        a_items = list(a_probabilities.items())
        y_items = list(y_probabilities.items())
        a_items = sorted(a_items)
        y_items = sorted(y_items)
        self._ret['a_items'] = a_items
        self._ret['y_items'] = y_items

        # map estimated values to original range and extract probabilities
        self._ret['mapped_values'] = [self.a_factory.value_to_estimation(
            a_item[0]) for a_item in self._ret['a_items']]
        self._ret['values'] = [a_item[0] for a_item in self._ret['a_items']]
        self._ret['y_values'] = [y_item[0] for y_item in y_items]
        self._ret['probabilities'] = [a_item[1]
                                      for a_item in self._ret['a_items']]
        self._ret['mapped_items'] = [(self._ret['mapped_values'][i], self._ret['probabilities'][i])
                                     for i in range(len(self._ret['mapped_values']))]

        # determine most likely estimator
        self._ret['value'] = None  # estimate in [0,1]
        self._ret['estimation'] = None  # estimate mapped to right interval
        self._ret['max_probability'] = 0
        for val, (est, prob) in zip(self._ret['values'], self._ret['mapped_items']):
            if prob > self._ret['max_probability']:
                self._ret['max_probability'] = prob
                self._ret['estimation'] = est
                self._ret['value'] = val

        # get MLE
        self._run_mle()

        # get 95% confidence interval
        alpha = 0.05
        kind = 'likelihood_ratio'  # empirically the most precise kind
        self._ret['95%_confidence_interval'] = self.confidence_interval(alpha, kind)

        return self._ret
