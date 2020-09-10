# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Quantum Phase Estimation-based Amplitude Estimation algorithm."""

from typing import Optional, Union, List, Tuple, Dict, Any
import logging
import warnings
from collections import OrderedDict
import numpy as np
from scipy.stats import chi2, norm
from scipy.optimize import bisect

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.utils import CircuitFactory
from qiskit.aqua.circuits import PhaseEstimationCircuit
from qiskit.aqua.utils.validation import validate_min
from .ae_algorithm import AmplitudeEstimationAlgorithm, AmplitudeEstimationAlgorithmResult
from .ae_utils import pdf_a, derivative_log_pdf_a, bisect_max

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class AmplitudeEstimation(AmplitudeEstimationAlgorithm):
    r"""The Quantum Phase Estimation-based Amplitude Estimation algorithm.

    This class implements the original Quantum Amplitude Estimation (QAE) algorithm, introduced by
    https://arxiv.org/abs/quant-ph/0005055. This (original) version uses quantum phase
    estimation along with a set of m ancilla qubits to find an estimate, that is restricted
    to the grid

        \{sin^2(\pi  y / 2^m) : y = 0, ..., 2^{m-1}\}.

    Using a maximum likelihood post processing, this grid constraint can be circumvented.
    This improved estimator is implemented as well, see https://arxiv.org/abs/1912.05559 Appendix A
    for more detail.
    """

    def __init__(self, num_eval_qubits: int,
                 a_factory: Optional[CircuitFactory] = None,
                 q_factory: Optional[CircuitFactory] = None,
                 i_objective: Optional[int] = None,
                 iqft: Optional[QuantumCircuit] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None) -> None:
        r"""
        Args:
            num_eval_qubits: Number of evaluation qubits, has a min. value of 1.
            a_factory: The CircuitFactory subclass object representing the problem unitary.
            q_factory: The CircuitFactory subclass object representing an amplitude estimation
                sample (based on a_factory).
            i_objective: The index of the objective qubit, i.e. the qubit marking 'good' solutions
                with the state \|1> and 'bad' solutions with the state \|0>.
            iqft: The Inverse Quantum Fourier Transform component, defaults to using a standard IQFT
                when None
            quantum_instance: Quantum Instance or Backend
        """
        validate_min('num_eval_qubits', num_eval_qubits, 1)
        super().__init__(a_factory, q_factory, i_objective, quantum_instance)

        # get parameters
        self._m = num_eval_qubits
        self._M = 2 ** num_eval_qubits
        self._iqft = iqft or QFT(self._m).inverse()
        self._circuit = None
        self._ret = {}  # type: Dict[str, Any]

    @property
    def _num_qubits(self) -> int:
        """Return the number of qubits needed in the circuit.

        Returns:
            The total number of qubits.
        """
        if self.a_factory is None:  # if A factory is not set, no qubits are specified
            return 0

        num_ancillas = self.q_factory.required_ancillas_controlled()
        num_qubits = self.a_factory.num_target_qubits + self._m + num_ancillas

        return num_qubits

    def construct_circuit(self, measurement: bool = False) -> QuantumCircuit:
        """Construct the Amplitude Estimation quantum circuit.

        Args:
            measurement: Boolean flag to indicate if measurements should be included in the circuit.

        Returns:
            The QuantumCircuit object for the constructed circuit.
        """
        pec = PhaseEstimationCircuit(
            iqft=self._iqft, num_ancillae=self._m,
            state_in_circuit_factory=self.a_factory,
            unitary_circuit_factory=self.q_factory
        )

        self._circuit = pec.construct_circuit(measurement=measurement)
        return self._circuit

    def _evaluate_statevector_results(self, probabilities: Union[List[float], np.ndarray]
                                      ) -> Tuple[OrderedDict, OrderedDict]:
        """Evaluate the results from statevector simulation.

        Given the probabilities from statevector simulation of the QAE circuit, compute the
        probabilities that the measurements y/gridpoints a are the best estimate.

        Args:
            probabilities: The probabilities obtained from the statevector simulation,
                i.e. real(statevector * statevector.conj())[0]

        Returns:
            Dictionaries containing the a gridpoints with respective probabilities and
                y measurements with respective probabilities, in this order.
        """
        # map measured results to estimates
        y_probabilities = OrderedDict()  # type: OrderedDict
        for i, probability in enumerate(probabilities):
            b = '{0:b}'.format(i).rjust(self._num_qubits, '0')[::-1]
            y = int(b[:self._m], 2)
            y_probabilities[y] = y_probabilities.get(y, 0) + probability

        a_probabilities = OrderedDict()  # type: OrderedDict
        for y, probability in y_probabilities.items():
            if y >= int(self._M / 2):
                y = self._M - y
            # due to the finite accuracy of the sine, we round the result to 7 decimals
            a = np.round(np.power(np.sin(y * np.pi / 2 ** self._m), 2),
                         decimals=7)
            a_probabilities[a] = a_probabilities.get(a, 0) + probability

        return a_probabilities, y_probabilities

    def _compute_fisher_information(self, observed: bool = False) -> float:
        """Computes the Fisher information for the output of the previous run.

        Args:
            observed: If True, the observed Fisher information is returned, otherwise
                the expected Fisher information.

        Returns:
            The Fisher information.
        """
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

    def _fisher_confint(self, alpha: float, observed: bool = False) -> List[float]:
        """Compute the Fisher information confidence interval for the MLE of the previous run.

        Args:
            alpha: Specifies the (1 - alpha) confidence level (0 < alpha < 1).
            observed: If True, the observed Fisher information is used to construct the
                confidence interval, otherwise the expected Fisher information.

        Returns:
            The Fisher information confidence interval.
        """
        shots = self._ret['shots']
        mle = self._ret['ml_value']

        # approximate the standard deviation of the MLE and construct the confidence interval
        std = np.sqrt(shots * self._compute_fisher_information(observed))
        ci = mle + norm.ppf(1 - alpha / 2) / std * np.array([-1, 1])

        # transform the confidence interval from [0, 1] to the target interval
        return [self.a_factory.value_to_estimation(bound) for bound in ci]

    def _likelihood_ratio_confint(self, alpha: float) -> List[float]:
        """Compute the likelihood ratio confidence interval for the MLE of the previous run.

        Args:
            alpha: Specifies the (1 - alpha) confidence level (0 < alpha < 1).

        Returns:
            The likelihood ratio confidence interval.
        """
        # Compute the two intervals in which we the look for values above
        # the likelihood ratio: the two bubbles next to the QAE estimate
        M = 2**self._m
        qae = self._ret['value']

        y = int(np.round(M * np.arcsin(np.sqrt(qae)) / np.pi))
        if y == 0:
            right_of_qae = np.sin(np.pi * (y + 1) / M)**2
            bubbles = [qae, right_of_qae]

        elif y == int(M / 2):  # remember, M = 2^m is a power of 2
            left_of_qae = np.sin(np.pi * (y - 1) / M)**2
            bubbles = [left_of_qae, qae]

        else:
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
        # It's valid to start off with the zero-width confidence interval, since the maximum
        # of the likelihood function is guaranteed to be over the threshold, and if alpha = 0
        # that's the valid interval
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

    def confidence_interval(self, alpha: float, kind: str = 'likelihood_ratio') -> List[float]:
        """Compute the (1 - alpha) confidence interval.

        Args:
            alpha: Confidence level: compute the (1 - alpha) confidence interval.
            kind: The method to compute the confidence interval, can be 'fisher', 'observed_fisher'
                or 'likelihood_ratio' (default)

        Returns:
            The (1 - alpha) confidence interval of the specified kind.

        Raises:
            AquaError: If 'mle' is not in self._ret.keys() (i.e. `run` was not called yet).
            NotImplementedError: If the confidence interval method `kind` is not implemented.
        """
        # check if AE did run already
        if 'mle' not in self._ret.keys():
            raise AquaError('Call run() first!')

        # if statevector simulator the estimate is exact
        if self._quantum_instance.is_statevector:
            return 2 * [self._ret['mle']]

        if kind in ['likelihood_ratio', 'lr']:
            return self._likelihood_ratio_confint(alpha)

        if kind in ['fisher', 'fi']:
            return self._fisher_confint(alpha, observed=False)

        if kind in ['observed_fisher', 'observed_information', 'oi']:
            return self._fisher_confint(alpha, observed=True)

        raise NotImplementedError('CI `{}` is not implemented.'.format(kind))

    def _run_mle(self) -> None:
        """Compute the Maximum Likelihood Estimator (MLE).

        Returns:
            The MLE for the previous AE run.

        Note:
            Before calling this method, call the method `run` of the AmplitudeEstimation instance.
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
        if y == 0:
            right_of_qae = np.sin(np.pi * (y + 1) / M)**2
            bubbles = [qae, right_of_qae]

        elif y == int(M / 2):  # remember, M = 2^m is a power of 2
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

    def _run(self) -> 'AmplitudeEstimationResult':
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
            y_probabilities = OrderedDict()
            a_probabilities = OrderedDict()
            shots = self._quantum_instance._run_config.shots

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

        # count the number of Q-oracle calls
        self._ret['num_oracle_queries'] = self._M - 1

        # get MLE
        self._run_mle()

        # get 95% confidence interval
        alpha = 0.05
        kind = 'likelihood_ratio'  # empirically the most precise kind
        self._ret['95%_confidence_interval'] = self.confidence_interval(alpha, kind)

        ae_result = AmplitudeEstimationAlgorithmResult()
        ae_result.a_estimation = self._ret['value']
        ae_result.estimation = self._ret['estimation']
        ae_result.num_oracle_queries = self._ret['num_oracle_queries']
        ae_result.confidence_interval = self._ret['95%_confidence_interval']

        result = AmplitudeEstimationResult()
        result.combine(ae_result)
        result.ml_value = self._ret['ml_value']
        result.mapped_a_samples = self._ret['values']
        result.probabilities = self._ret['probabilities']
        result.shots = self._ret['shots']
        result.mle = self._ret['mle']
        if 'statevector' in self._ret:
            result.circuit_result = self._ret['statevector']
        elif 'counts' in self._ret:
            result.circuit_result = dict(self._ret['counts'])
        result.a_samples = self._ret['a_items']
        result.y_measurements = self._ret['y_items']
        result.mapped_values = self._ret['mapped_values']
        result.max_probability = self._ret['max_probability']
        return result


class AmplitudeEstimationResult(AmplitudeEstimationAlgorithmResult):
    """ AmplitudeEstimation Result."""

    @property
    def ml_value(self) -> float:
        """ returns ml_value """
        return self.get('ml_value')

    @ml_value.setter
    def ml_value(self, value: float) -> None:
        """ set ml_value """
        self.data['ml_value'] = value

    @property
    def mapped_a_samples(self) -> List[float]:
        """ return mapped_a_samples  """
        return self.get('mapped_a_samples')

    @mapped_a_samples.setter
    def mapped_a_samples(self, value: List[float]) -> None:
        """ set mapped_a_samples """
        self.data['mapped_a_samples'] = value

    @property
    def probabilities(self) -> List[float]:
        """ return probabilities """
        return self.get('probabilities')

    @probabilities.setter
    def probabilities(self, value: List[float]) -> None:
        """ set probabilities """
        self.data['probabilities'] = value

    @property
    def shots(self) -> int:
        """ return shots """
        return self.get('shots')

    @shots.setter
    def shots(self, value: int) -> None:
        """ set shots """
        self.data['shots'] = value

    @property
    def mle(self) -> float:
        """ return mle """
        return self.get('mle')

    @mle.setter
    def mle(self, value: float) -> None:
        """ set mle """
        self.data['mle'] = value

    @property
    def circuit_result(self) -> Optional[Union[np.ndarray, Dict[str, int]]]:
        """ return circuit result """
        return self.get('circuit_result')

    @circuit_result.setter
    def circuit_result(self, value: Union[np.ndarray, Dict[str, int]]) -> None:
        """ set circuit result """
        self.data['circuit_result'] = value

    @property
    def a_samples(self) -> List[Tuple[float, float]]:
        """ return a_samples """
        return self.get('a_samples')

    @a_samples.setter
    def a_samples(self, value: List[Tuple[float, float]]) -> None:
        """ set a_samples """
        self.data['a_samples'] = value

    @property
    def y_measurements(self) -> List[Tuple[int, float]]:
        """ return y_measurements """
        return self.get('y_measurements')

    @y_measurements.setter
    def y_measurements(self, value: List[Tuple[int, float]]) -> None:
        """ set y_measurements """
        self.data['y_measurements'] = value

    @property
    def mapped_values(self) -> List[float]:
        """ return mapped_values """
        return self.get('mapped_values')

    @mapped_values.setter
    def mapped_values(self, value: List[float]) -> None:
        """ set mapped_values """
        self.data['mapped_values'] = value

    @property
    def max_probability(self) -> float:
        """ return max_probability """
        return self.get('max_probability')

    @max_probability.setter
    def max_probability(self, value: float) -> None:
        """ set max_probability """
        self.data['max_probability'] = value

    @staticmethod
    def from_dict(a_dict: Dict) -> 'AmplitudeEstimationResult':
        """ create new object from a dictionary """
        return AmplitudeEstimationResult(a_dict)

    def __getitem__(self, key: object) -> object:
        if key == 'statevector':
            warnings.warn('statevector deprecated, use circuit_result property.',
                          DeprecationWarning)
            return super().__getitem__('circuit_result')
        elif key == 'counts':
            warnings.warn('counts deprecated, use circuit_result property.', DeprecationWarning)
            return super().__getitem__('circuit_result')
        elif key == 'values':
            warnings.warn('values deprecated, use mapped_a_samples property.', DeprecationWarning)
            return super().__getitem__('mapped_a_samples')
        elif key == 'y_items':
            warnings.warn('y_items deprecated, use y_measurements property.', DeprecationWarning)
            return super().__getitem__('y_measurements')
        elif key == 'a_items':
            warnings.warn('a_items deprecated, use a_samples property.', DeprecationWarning)
            return super().__getitem__('a_samples')

        return super().__getitem__(key)
