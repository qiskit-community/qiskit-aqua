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

"""The Maximum Likelihood Amplitude Estimation algorithm."""

from typing import Optional, List, Union, Tuple, Callable, Dict, Any
import warnings
import logging
import numpy as np
from scipy.optimize import brute
from scipy.stats import norm, chi2

from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.utils.circuit_factory import CircuitFactory
from qiskit.aqua.utils.validation import validate_min
from .ae_algorithm import AmplitudeEstimationAlgorithm, AmplitudeEstimationAlgorithmResult

logger = logging.getLogger(__name__)


class MaximumLikelihoodAmplitudeEstimation(AmplitudeEstimationAlgorithm):
    """The Maximum Likelihood Amplitude Estimation algorithm.

    This class implements the quantum amplitude estimation (QAE) algorithm without phase
    estimation, as introduced in [1]. In comparison to the original QAE algorithm [2],
    this implementation relies solely on different powers of the Grover operator and does not
    require additional evaluation qubits.
    Finally, the estimate is determined via a maximum likelihood estimation, which is why this
    class in named ``MaximumLikelihoodAmplitudeEstimation``.

    References:
        [1]: Suzuki, Y., Uno, S., Raymond, R., Tanaka, T., Onodera, T., & Yamamoto, N. (2019).
             Amplitude Estimation without Phase Estimation.
             `arXiv:1904.10246 <https://arxiv.org/abs/1904.10246>`_.
        [2]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
             Quantum Amplitude Amplification and Estimation.
             `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.
    """

    def __init__(self, num_oracle_circuits: int,
                 state_preparation: Optional[Union[QuantumCircuit, CircuitFactory]] = None,
                 grover_operator: Optional[Union[QuantumCircuit, CircuitFactory]] = None,
                 objective_qubits: Optional[List[int]] = None,
                 post_processing: Optional[Callable[[float], float]] = None,
                 a_factory: Optional[CircuitFactory] = None,
                 q_factory: Optional[CircuitFactory] = None,
                 i_objective: Optional[int] = None,
                 likelihood_evals: Optional[int] = None,
                 quantum_instance: Optional[
                     Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:
        r"""
        Args:
            num_oracle_circuits: The number of circuits applying different powers of the Grover
                oracle Q. The (`num_oracle_circuits` + 1) executed circuits will be
                `[id, Q^2^0, ..., Q^2^{num_oracle_circuits-1}] A |0>`, where A is the problem
                unitary encoded in the argument `a_factory`.
                Has a minimum value of 1.
            state_preparation: A circuit preparing the input state, referred to as
                :math:`\mathcal{A}`.
            grover_operator: The Grover operator :math:`\mathcal{Q}` used as unitary in the
                phase estimation circuit.
            objective_qubits: A list of qubit indices. A measurement outcome is classified as
                'good' state if all objective qubits are in state :math:`|1\rangle`, otherwise it
                is classified as 'bad'.
            post_processing: A mapping applied to the estimate of :math:`0 \leq a \leq 1`,
                usually used to map the estimate to a target interval.
            a_factory: The CircuitFactory subclass object representing the problem unitary.
            q_factory: The CircuitFactory subclass object representing.
                an amplitude estimation sample (based on a_factory)
            i_objective: The index of the objective qubit, i.e. the qubit marking 'good' solutions
                with the state \|1> and 'bad' solutions with the state \|0>
            likelihood_evals: The number of gridpoints for the maximum search of the likelihood
                function
            quantum_instance: Quantum Instance or Backend
        """
        validate_min('num_oracle_circuits', num_oracle_circuits, 1)

        # support legacy input if passed as positional arguments
        if isinstance(state_preparation, CircuitFactory):
            a_factory = state_preparation
            state_preparation = None

        if isinstance(grover_operator, CircuitFactory):
            q_factory = grover_operator
            grover_operator = None

        if isinstance(objective_qubits, int):
            i_objective = objective_qubits
            objective_qubits = None

        super().__init__(state_preparation=state_preparation,
                         grover_operator=grover_operator,
                         objective_qubits=objective_qubits,
                         post_processing=post_processing,
                         a_factory=a_factory,
                         q_factory=q_factory,
                         i_objective=i_objective,
                         quantum_instance=quantum_instance)

        # get parameters
        self._evaluation_schedule = [0] + [2**j for j in range(num_oracle_circuits)]

        self._likelihood_evals = likelihood_evals
        # default number of evaluations is max(10^5, pi/2 * 10^3 * 2^(m))
        if likelihood_evals is None:
            default = 10000
            self._likelihood_evals = np.maximum(default,
                                                int(np.pi / 2 * 1000 * 2 ** num_oracle_circuits))

        self._circuits = []  # type: List[QuantumCircuit]
        self._ret = {}  # type: Dict[str, Any]

    def construct_circuits(self, measurement: bool = False) -> List[QuantumCircuit]:
        """Construct the Amplitude Estimation w/o QPE quantum circuits.

        Args:
            measurement: Boolean flag to indicate if measurement should be included in the circuits.

        Returns:
            A list with the QuantumCircuit objects for the algorithm.
        """
        # keep track of the Q-oracle queries
        self._ret['num_oracle_queries'] = 0
        self._circuits = []

        if self.state_preparation is not None:   # using circuits, not CircuitFactory
            num_qubits = max(self.state_preparation.num_qubits, self.grover_operator.num_qubits)
            q = QuantumRegister(num_qubits, 'q')
            qc_0 = QuantumCircuit(q, name='qc_a')  # 0 applications of Q, only a single A operator

            # add classical register if needed
            if measurement:
                c = ClassicalRegister(len(self.objective_qubits))
                qc_0.add_register(c)

            qc_0.compose(self.state_preparation, inplace=True)

            for k in self._evaluation_schedule:
                qc_k = qc_0.copy(name='qc_a_q_%s' % k)

                if k != 0:
                    qc_k.compose(self.grover_operator.power(k), inplace=True)

                if measurement:
                    # real hardware can currently not handle operations after measurements,
                    # which might happen if the circuit gets transpiled, hence we're adding
                    # a safeguard-barrier
                    qc_k.barrier()
                    qc_k.measure(self.objective_qubits, *c)

                self._circuits += [qc_k]
        else:  # using deprecated CircuitFactory
            # construct first part of circuit
            q = QuantumRegister(self._a_factory.num_target_qubits, 'q')
            qc_0 = QuantumCircuit(q, name='qc_a')  # 0 applications of Q, only a single A operator

            warnings.filterwarnings('ignore', category=DeprecationWarning)
            q_factory = self.q_factory
            warnings.filterwarnings('always', category=DeprecationWarning)

            # get number of ancillas
            num_ancillas = np.maximum(self._a_factory.required_ancillas(),
                                      q_factory.required_ancillas())

            q_aux = None
            # pylint: disable=comparison-with-callable
            if num_ancillas > 0:
                q_aux = QuantumRegister(num_ancillas, 'aux')
                qc_0.add_register(q_aux)

            # add classical register if needed
            if measurement:
                c = ClassicalRegister(len(self.objective_qubits))
                qc_0.add_register(c)

            self._a_factory.build(qc_0, q, q_aux)

            for k in self._evaluation_schedule:
                qc_k = qc_0.copy(name='qc_a_q_%s' % k)

                if k != 0:
                    q_factory.build_power(qc_k, q, k, q_aux)

                if measurement:
                    # real hardware can currently not handle operations after measurements,
                    # which might happen if the circuit gets transpiled, hence we're adding
                    # a safeguard-barrier
                    qc_k.barrier()
                    qc_k.measure([q[obj] for obj in self.objective_qubits], c)

                self._circuits += [qc_k]

        return self._circuits

    def _evaluate_statevectors(self,
                               statevectors: Union[List[List[complex]], List[np.ndarray]]
                               ) -> List[float]:
        """For each statevector compute the probability that |1> is measured in the objective qubit.

        Args:
            statevectors: A list of statevectors.

        Raises:
            AquaError: If `construct_circuit` has not been called before. The `construct_circuit`
                method sets an internal variable required in this method.

        Returns:
            The corresponding probabilities.
        """
        if self._circuits is None:
            raise AquaError('Before calling _evaluate_statevector_results the construct_circuit '
                            'method must be called, which sets the internal _circuit variable '
                            'required in this method.')

        num_qubits = self._circuits[0].num_qubits

        probabilities = []
        for statevector in statevectors:
            p_k = 0.0
            for i, amplitude in enumerate(statevector):
                probability = np.abs(amplitude) ** 2
                bitstr = ('{:0%db}' % num_qubits).format(i)[::-1]
                if self.is_good_state(bitstr):
                    p_k += probability
            probabilities += [p_k]

        return probabilities

    def _get_hits(self) -> Tuple[List[float], List[int]]:
        """Get the good and total counts.

        Returns:
            A pair of two lists, ([1-counts per experiment], [shots per experiment]).

        Raises:
            AquaError: If self.run() has not been called yet.
        """
        one_hits = []  # h_k: how often 1 has been measured, for a power Q^(m_k)
        all_hits: List = []  # shots_k: how often has been measured at a power Q^(m_k)
        try:
            if self.quantum_instance.is_statevector:
                probabilities = self._evaluate_statevectors(self._ret['statevectors'])
                one_hits = probabilities
                all_hits = np.ones_like(one_hits)  # type: ignore
            else:
                for c in self._ret['counts']:
                    one_hits += [c.get('1', 0)]  # return 0 if no key '1' found
                    all_hits += [sum(c.values())]
        except KeyError as ex:
            raise AquaError('Call run() first!') from ex

        return one_hits, all_hits

    def _safe_min(self, array, default=0):
        if len(array) == 0:
            return default
        return np.min(array)

    def _safe_max(self, array, default=(np.pi / 2)):
        if len(array) == 0:
            return default
        return np.max(array)

    def _compute_fisher_information(self, a: Optional[float] = None,
                                    num_sum_terms: Optional[int] = None,
                                    observed: bool = False) -> float:
        """Compute the Fisher information.

        Args:
            a: The amplitude `a`. Can be omitted if `run` was called already, then the
                estimate of the algorithm is used.
            num_sum_terms: The number of sum terms to be included in the calculation of the
                Fisher information. By default all values are included.
            observed: If True, compute the observed Fisher information, otherwise the theoretical
                one.

        Returns:
            The computed Fisher information, or np.inf if statevector simulation was used.

        Raises:
            KeyError: Call run() first!
        """
        # Set the value a. Use `est_a` if provided.
        if a is None:
            try:
                a = self._ret['value']
            except KeyError as ex:
                raise KeyError('Call run() first!') from ex

        # Corresponding angle to the value a (only use real part of 'a')
        theta_a = np.arcsin(np.sqrt(np.real(a)))

        # Get the number of hits (shots_k) and one-hits (h_k)
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
            # Note, that the observed Fisher information is very unreliable in this algorithm!
            d_loglik = 0
            for shots_k, h_k, m_k in zip(all_hits, one_hits, evaluation_schedule):
                tan = np.tan((2 * m_k + 1) * theta_a)
                d_loglik += (2 * m_k + 1) * (h_k / tan + (shots_k - h_k) * tan)

            d_loglik /= np.sqrt(a * (1 - a))
            fisher_information = d_loglik ** 2 / len(all_hits)

        else:
            fisher_information = sum(shots_k * (2 * m_k + 1)**2
                                     for shots_k, m_k in zip(all_hits, evaluation_schedule))
            fisher_information /= a * (1 - a)

        return fisher_information

    def _fisher_confint(self, alpha: float = 0.05, observed: bool = False) -> List[float]:
        """Compute the `alpha` confidence interval based on the Fisher information.

        Args:
            alpha: The level of the confidence interval (must be <= 0.5), default to 0.05.
            observed: If True, use observed Fisher information.

        Returns:
            float: The alpha confidence interval based on the Fisher information
        Raises:
            AssertionError: Call run() first!
        """
        # Get the (observed) Fisher information
        fisher_information = None
        try:
            fisher_information = self._ret['fisher_information']
        except KeyError as ex:
            raise AssertionError("Call run() first!") from ex

        if observed:
            fisher_information = self._compute_fisher_information(observed=True)

        normal_quantile = norm.ppf(1 - alpha / 2)
        confint = np.real(self._ret['value']) + \
            normal_quantile / np.sqrt(fisher_information) * np.array([-1, 1])
        mapped_confint = [self.post_processing(bound) for bound in confint]
        return mapped_confint

    def _likelihood_ratio_confint(self, alpha: float = 0.05,
                                  nevals: Optional[int] = None) -> List[float]:
        """Compute the likelihood-ratio confidence interval.

        Args:
            alpha: The level of the confidence interval (< 0.5), defaults to 0.05.
            nevals: The number of evaluations to find the intersection with the loglikelihood
                function. Defaults to an adaptive value based on the maximal power of Q.

        Returns:
            The alpha-likelihood-ratio confidence interval.
        """
        if nevals is None:
            nevals = self._likelihood_evals

        def loglikelihood(theta, one_counts, all_counts):
            loglik = 0
            for i, k in enumerate(self._evaluation_schedule):
                loglik += np.log(np.sin((2 * k + 1) * theta) ** 2) * one_counts[i]
                loglik += np.log(np.cos((2 * k + 1) * theta) ** 2) * (all_counts[i] - one_counts[i])
            return loglik

        one_counts, all_counts = self._get_hits()

        eps = 1e-15  # to avoid invalid value in log
        thetas = np.linspace(0 + eps, np.pi / 2 - eps, nevals)
        values = np.zeros(len(thetas))
        for i, theta in enumerate(thetas):
            values[i] = loglikelihood(theta, one_counts, all_counts)

        loglik_mle = loglikelihood(self._ret['theta'], one_counts, all_counts)
        chi2_quantile = chi2.ppf(1 - alpha, df=1)
        thres = loglik_mle - chi2_quantile / 2

        # the (outer) LR confidence interval
        above_thres = thetas[values >= thres]

        # it might happen that the `above_thres` array is empty,
        # to still provide a valid result use safe_min/max which
        # then yield [0, pi/2]
        confint = [self._safe_min(above_thres, default=0),
                   self._safe_max(above_thres, default=np.pi / 2)]
        mapped_confint = [self.post_processing(np.sin(bound) ** 2) for bound in confint]

        return mapped_confint

    def confidence_interval(self, alpha: float, kind: str = 'fisher') -> List[float]:
        # pylint: disable=wrong-spelling-in-docstring
        """Compute the `alpha` confidence interval using the method `kind`.

        The confidence level is (1 - `alpha`) and supported kinds are 'fisher',
        'likelihood_ratio' and 'observed_fisher' with shorthand
        notations 'fi', 'lr' and 'oi', respectively.

        Args:
            alpha: The confidence level.
            kind: The method to compute the confidence interval. Defaults to 'fisher', which
                computes the theoretical Fisher information.

        Returns:
            The specified confidence interval.

        Raises:
            AquaError: If `run()` hasn't been called yet.
            NotImplementedError: If the method `kind` is not supported.
        """
        # check if AE did run already
        if 'estimation' not in self._ret.keys():
            raise AquaError('Call run() first!')

        # if statevector simulator the estimate is exact
        if self._quantum_instance.is_statevector:
            return 2 * [self._ret['estimation']]

        if kind in ['likelihood_ratio', 'lr']:
            return self._likelihood_ratio_confint(alpha)

        if kind in ['fisher', 'fi']:
            return self._fisher_confint(alpha, observed=False)

        if kind in ['observed_fisher', 'observed_information', 'oi']:
            return self._fisher_confint(alpha, observed=True)

        raise NotImplementedError('CI `{}` is not implemented.'.format(kind))

    def _compute_mle_safe(self):
        """Compute the MLE via a grid-search.

        This is a stable approach if sufficient gridpoints are used.
        """
        one_hits, all_hits = self._get_hits()

        # search range
        eps = 1e-15  # to avoid invalid value in log
        search_range = [0 + eps, np.pi / 2 - eps]

        def loglikelihood(theta):
            # loglik contains the first `it` terms of the full loglikelihood
            loglik = 0
            for i, k in enumerate(self._evaluation_schedule):
                loglik += np.log(np.sin((2 * k + 1) * theta) ** 2) * one_hits[i]
                loglik += np.log(np.cos((2 * k + 1) * theta) ** 2) * (all_hits[i] - one_hits[i])
            return -loglik

        est_theta = brute(loglikelihood, [search_range], Ns=self._likelihood_evals)[0]
        return est_theta

    def _run_mle(self) -> float:
        """Compute the maximum likelihood estimator (MLE) for the angle theta.

        Returns:
            The MLE for the angle theta, related to the amplitude a via a = sin^2(theta)
        """
        # TODO implement a **reliable**, fast method to find the maximum of the likelihood function
        return self._compute_mle_safe()

    def _run(self) -> 'MaximumLikelihoodAmplitudeEstimationResult':
        # check if A factory or state_preparation has been set
        if self.state_preparation is None:
            if self._a_factory is None:  # getter emits deprecation warnings, therefore nest
                raise AquaError('Either the state_preparation variable or the a_factory '
                                '(deprecated) must be set to run the algorithm.')

        if self._quantum_instance.is_statevector:

            # run circuit on statevector simulator
            self.construct_circuits(measurement=False)
            ret = self._quantum_instance.execute(self._circuits)

            # get statevectors and construct MLE input
            statevectors = [np.asarray(ret.get_statevector(circuit)) for circuit in self._circuits]
            self._ret['statevectors'] = statevectors

            # to count the number of Q-oracle calls (don't count shots)
            shots = 1

        else:
            # run circuit on QASM simulator
            self.construct_circuits(measurement=True)
            ret = self._quantum_instance.execute(self._circuits)

            # get counts and construct MLE input
            self._ret['counts'] = [ret.get_counts(circuit) for circuit in self._circuits]

            # to count the number of Q-oracle calls
            shots = self._quantum_instance._run_config.shots

        # run maximum likelihood estimation and construct results
        self._ret['theta'] = self._run_mle()
        self._ret['value'] = np.sin(self._ret['theta'])**2
        self._ret['estimation'] = self.post_processing(self._ret['value'])
        self._ret['fisher_information'] = self._compute_fisher_information()
        self._ret['num_oracle_queries'] = shots * sum(k for k in self._evaluation_schedule)

        confidence_interval = self._fisher_confint(alpha=0.05)
        self._ret['95%_confidence_interval'] = confidence_interval

        ae_result = AmplitudeEstimationAlgorithmResult()
        ae_result.a_estimation = self._ret['value']
        ae_result.estimation = self._ret['estimation']
        ae_result.num_oracle_queries = self._ret['num_oracle_queries']
        ae_result.confidence_interval = self._ret['95%_confidence_interval']

        result = MaximumLikelihoodAmplitudeEstimationResult()
        result.combine(ae_result)
        if 'statevectors' in self._ret:
            result.circuit_results = self._ret['statevectors']
        elif 'counts' in self._ret:
            result.circuit_results = self._ret['counts']
        result.theta = self._ret['theta']
        result.fisher_information = self._ret['fisher_information']
        return result


class MaximumLikelihoodAmplitudeEstimationResult(AmplitudeEstimationAlgorithmResult):
    """ MaximumLikelihoodAmplitudeEstimation Result."""

    @property
    def circuit_results(self) -> Optional[Union[List[np.ndarray], List[Dict[str, int]]]]:
        """ return circuit results """
        return self.get('circuit_results')

    @circuit_results.setter
    def circuit_results(self, value: Union[List[np.ndarray], List[Dict[str, int]]]) -> None:
        """ set circuit results """
        self.data['circuit_results'] = value

    @property
    def theta(self) -> float:
        """ returns theta """
        return self.get('theta')

    @theta.setter
    def theta(self, value: float) -> None:
        """ set theta """
        self.data['theta'] = value

    @property
    def fisher_information(self) -> float:
        """ return fisher_information  """
        return self.get('fisher_information')

    @fisher_information.setter
    def fisher_information(self, value: float) -> None:
        """ set fisher_information """
        self.data['fisher_information'] = value

    @staticmethod
    def from_dict(a_dict: Dict) -> 'MaximumLikelihoodAmplitudeEstimationResult':
        """ create new object from a dictionary """
        return MaximumLikelihoodAmplitudeEstimationResult(a_dict)

    def __getitem__(self, key: object) -> object:
        if key == 'statevectors':
            warnings.warn('statevectors deprecated, use circuit_results property.',
                          DeprecationWarning)
            return super().__getitem__('circuit_results')
        elif key == 'counts':
            warnings.warn('counts deprecated, use circuit_results property.', DeprecationWarning)
            return super().__getitem__('circuit_results')

        return super().__getitem__(key)
