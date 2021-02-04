
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

"""The Iterative Quantum Amplitude Estimation Algorithm."""

import warnings
from typing import Optional, Union, List, Tuple, Callable, Dict, Any, cast
import logging
import numpy as np
from scipy.stats import beta

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.utils.circuit_factory import CircuitFactory
from qiskit.aqua.utils.validation import validate_range, validate_in_set

from .ae_algorithm import AmplitudeEstimationAlgorithm, AmplitudeEstimationAlgorithmResult

logger = logging.getLogger(__name__)


class IterativeAmplitudeEstimation(AmplitudeEstimationAlgorithm):
    r"""The Iterative Amplitude Estimation algorithm.

    This class implements the Iterative Quantum Amplitude Estimation (IQAE) algorithm, proposed
    in [1]. The output of the algorithm is an estimate that,
    with at least probability :math:`1 - \alpha`, differs by epsilon to the target value, where
    both alpha and epsilon can be specified.

    It differs from the original QAE algorithm proposed by Brassard [2] in that it does not rely on
    Quantum Phase Estimation, but is only based on Grover's algorithm. IQAE iteratively
    applies carefully selected Grover iterations to find an estimate for the target amplitude.

    References:
        [1]: Grinko, D., Gacon, J., Zoufal, C., & Woerner, S. (2019).
             Iterative Quantum Amplitude Estimation.
             `arXiv:1912.05559 <https://arxiv.org/abs/1912.05559>`_.
        [2]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
             Quantum Amplitude Amplification and Estimation.
             `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.
    """

    def __init__(self, epsilon: float,
                 alpha: float,
                 confint_method: str = 'beta',
                 min_ratio: float = 2,
                 state_preparation: Optional[Union[QuantumCircuit, CircuitFactory]] = None,
                 grover_operator: Optional[Union[QuantumCircuit, CircuitFactory]] = None,
                 objective_qubits: Optional[List[int]] = None,
                 post_processing: Optional[Callable[[float], float]] = None,
                 a_factory: Optional[CircuitFactory] = None,
                 q_factory: Optional[CircuitFactory] = None,
                 i_objective: Optional[int] = None,
                 initial_state: Optional[QuantumCircuit] = None,
                 quantum_instance: Optional[
                     Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:
        r"""
        The output of the algorithm is an estimate for the amplitude `a`, that with at least
        probability 1 - alpha has an error of epsilon. The number of A operator calls scales
        linearly in 1/epsilon (up to a logarithmic factor).

        Args:
            epsilon: Target precision for estimation target `a`, has values between 0 and 0.5
            alpha: Confidence level, the target probability is 1 - alpha, has values between 0 and 1
            confint_method: Statistical method used to estimate the confidence intervals in
                each iteration, can be 'chernoff' for the Chernoff intervals or 'beta' for the
                Clopper-Pearson intervals (default)
            min_ratio: Minimal q-ratio (:math:`K_{i+1} / K_i`) for FindNextK
            state_preparation: A circuit preparing the input state, referred to as
                :math:`\mathcal{A}`.
            grover_operator: The Grover operator :math:`\mathcal{Q}` used as unitary in the
                phase estimation circuit.
            objective_qubits: A list of qubit indices. A measurement outcome is classified as
                'good' state if all objective qubits are in state :math:`|1\rangle`, otherwise it
                is classified as 'bad'.
            post_processing: A mapping applied to the estimate of :math:`0 \leq a \leq 1`,
                usually used to map the estimate to a target interval.
            a_factory: The A operator, specifying the QAE problem
            q_factory: The Q operator (Grover operator), constructed from the
                A operator
            i_objective: Index of the objective qubit, that marks the 'good/bad' states
            initial_state: A state to prepend to the constructed circuits.
            quantum_instance: Quantum Instance or Backend

        Raises:
            AquaError: if the method to compute the confidence intervals is not supported
        """
        # validate ranges of input arguments
        validate_range('epsilon', epsilon, 0, 0.5)
        validate_range('alpha', alpha, 0, 1)
        validate_in_set('confint_method', confint_method, {'chernoff', 'beta'})

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

        # store parameters
        self._epsilon = epsilon
        self._alpha = alpha
        self._min_ratio = min_ratio
        self._confint_method = confint_method
        self._initial_state = initial_state

        # results dictionary
        self._ret = {}  # type: Dict[str, Any]

    @property
    def precision(self) -> float:
        """Returns the target precision `epsilon` of the algorithm.

        Returns:
            The target precision (which is half the width of the confidence interval).
        """
        return self._epsilon

    @precision.setter
    def precision(self, epsilon: float) -> None:
        """Set the target precision of the algorithm.

        Args:
            epsilon: Target precision for estimation target `a`.
        """
        self._epsilon = epsilon

    def _find_next_k(self, k: int, upper_half_circle: bool, theta_interval: Tuple[float, float],
                     min_ratio: float = 2.0) -> Tuple[int, bool]:
        """Find the largest integer k_next, such that the interval (4 * k_next + 2)*theta_interval
        lies completely in [0, pi] or [pi, 2pi], for theta_interval = (theta_lower, theta_upper).

        Args:
            k: The current power of the Q operator.
            upper_half_circle: Boolean flag of whether theta_interval lies in the
                upper half-circle [0, pi] or in the lower one [pi, 2pi].
            theta_interval: The current confidence interval for the angle theta,
                i.e. (theta_lower, theta_upper).
            min_ratio: Minimal ratio K/K_next allowed in the algorithm.

        Returns:
            The next power k, and boolean flag for the extrapolated interval.

        Raises:
            AquaError: if min_ratio is smaller or equal to 1
        """
        if min_ratio <= 1:
            raise AquaError('min_ratio must be larger than 1 to ensure convergence')

        # initialize variables
        theta_l, theta_u = theta_interval
        old_scaling = 4 * k + 2  # current scaling factor, called K := (4k + 2)

        # the largest feasible scaling factor K cannot be larger than K_max,
        # which is bounded by the length of the current confidence interval
        max_scaling = int(1 / (2 * (theta_u - theta_l)))
        scaling = max_scaling - (max_scaling - 2) % 4  # bring into the form 4 * k_max + 2

        # find the largest feasible scaling factor K_next, and thus k_next
        while scaling >= min_ratio * old_scaling:
            theta_min = scaling * theta_l - int(scaling * theta_l)
            theta_max = scaling * theta_u - int(scaling * theta_u)

            if theta_min <= theta_max <= 0.5 and theta_min <= 0.5:
                # the extrapolated theta interval is in the upper half-circle
                upper_half_circle = True
                return int((scaling - 2) / 4), upper_half_circle

            elif theta_max >= 0.5 and theta_max >= theta_min >= 0.5:
                # the extrapolated theta interval is in the upper half-circle
                upper_half_circle = False
                return int((scaling - 2) / 4), upper_half_circle

            scaling -= 4

        # if we do not find a feasible k, return the old one
        return int(k), upper_half_circle

    def construct_circuit(self, k: int, measurement: bool = False) -> QuantumCircuit:
        r"""Construct the circuit Q^k A \|0>.

        The A operator is the unitary specifying the QAE problem and Q the associated Grover
        operator.

        Args:
            k: The power of the Q operator.
            measurement: Boolean flag to indicate if measurements should be included in the
                circuits.

        Returns:
            The circuit Q^k A \|0>.
        """
        if self.state_preparation is not None:   # using circuits, not CircuitFactory
            num_qubits = max(self.state_preparation.num_qubits, self.grover_operator.num_qubits)
            circuit = QuantumCircuit(num_qubits, name='circuit')

            if self._initial_state is not None:
                circuit.compose(self._initial_state, inplace=True)

            # add classical register if needed
            if measurement:
                c = ClassicalRegister(len(self.objective_qubits))
                circuit.add_register(c)

            # add A operator
            circuit.compose(self.state_preparation, inplace=True)

            # add Q^k
            if k != 0:
                circuit.compose(self.grover_operator.power(k), inplace=True)
        else:  # deprecated CircuitFactory
            q = QuantumRegister(self._a_factory.num_target_qubits, 'q')
            circuit = QuantumCircuit(q, name='circuit')

            warnings.filterwarnings('ignore', category=DeprecationWarning)
            q_factory = self.q_factory
            warnings.filterwarnings('always', category=DeprecationWarning)

            # get number of ancillas and add register if needed
            num_ancillas = np.maximum(self._a_factory.required_ancillas(),
                                      q_factory.required_ancillas())

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
            self._a_factory.build(circuit, q, q_aux)

            # add Q^k
            if k != 0:
                q_factory.build_power(circuit, q, k, q_aux)

            # add optional measurement
        if measurement:
            # real hardware can currently not handle operations after measurements, which might
            # happen if the circuit gets transpiled, hence we're adding a safeguard-barrier
            circuit.barrier()
            circuit.measure(self.objective_qubits, *c)

        return circuit

    def _probability_to_measure_one(self,
                                    counts_or_statevector: Union[dict, List[complex], np.ndarray]
                                    ) -> Union[Tuple[int, float], float]:
        """Get the probability to measure '1' in the last qubit.

        Args:
            counts_or_statevector: Either a counts-dictionary (with one measured qubit only!) or
                the statevector returned from the statevector_simulator.

        Returns:
            If a dict is given, return (#one-counts, #one-counts/#all-counts),
            otherwise Pr(measure '1' in the last qubit).
        """
        if self.state_preparation is not None:
            num_qubits = self.state_preparation.num_qubits - self.state_preparation.num_ancillas
        else:
            num_qubits = self._a_factory.num_target_qubits

        if isinstance(counts_or_statevector, dict):
            one_counts = counts_or_statevector.get('1' * len(self.objective_qubits), 0)
            return int(one_counts), one_counts / sum(counts_or_statevector.values())
        else:
            statevector = counts_or_statevector

            # sum over all amplitudes where the objective qubit is 1
            prob = 0
            for i, amplitude in enumerate(statevector):
                bitstr = ('{:0%db}' % num_qubits).format(i)[::-1]
                if self.is_good_state(bitstr):
                    prob = prob + np.abs(amplitude)**2

            return prob

    def _chernoff_confint(self, value: float, shots: int, max_rounds: int, alpha: float
                          ) -> Tuple[float, float]:
        """Compute the Chernoff confidence interval for `shots` i.i.d. Bernoulli trials.

        The confidence interval is

            [value - eps, value + eps], where eps = sqrt(3 * log(2 * max_rounds/ alpha) / shots)

        but at most [0, 1].

        Args:
            value: The current estimate.
            shots: The number of shots.
            max_rounds: The maximum number of rounds, used to compute epsilon_a.
            alpha: The confidence level, used to compute epsilon_a.

        Returns:
            The Chernoff confidence interval.
        """
        eps = np.sqrt(3 * np.log(2 * max_rounds / alpha) / shots)
        lower = np.maximum(0, value - eps)
        upper = np.minimum(1, value + eps)
        return lower, upper

    def _clopper_pearson_confint(self, counts: int, shots: int, alpha: float
                                 ) -> Tuple[float, float]:
        """Compute the Clopper-Pearson confidence interval for `shots` i.i.d. Bernoulli trials.

        Args:
            counts: The number of positive counts.
            shots: The number of shots.
            alpha: The confidence level for the confidence interval.

        Returns:
            The Clopper-Pearson confidence interval.
        """
        lower, upper = 0, 1

        # if counts == 0, the beta quantile returns nan
        if counts != 0:
            lower = beta.ppf(alpha / 2, counts, shots - counts + 1)

        # if counts == shots, the beta quantile returns nan
        if counts != shots:
            upper = beta.ppf(1 - alpha / 2, counts + 1, shots - counts)

        return lower, upper

    def _run(self) -> 'IterativeAmplitudeEstimationResult':
        # check if A factory or state_preparation has been set
        if self.state_preparation is None:
            if self._a_factory is None:  # getter emits deprecation warnings, therefore nest
                raise AquaError('Either the state_preparation variable or the a_factory '
                                '(deprecated) must be set to run the algorithm.')

        # initialize memory variables
        powers = [0]  # list of powers k: Q^k, (called 'k' in paper)
        ratios = []  # list of multiplication factors (called 'q' in paper)
        theta_intervals = [[0, 1 / 4]]  # a priori knowledge of theta / 2 / pi
        a_intervals = [[0.0, 1.0]]  # a priori knowledge of the confidence interval of the estimate
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
            prob = cast(float, prob)  # tell MyPy it's a float and not Tuple[int, float ]

            a_confidence_interval = [prob, prob]  # type: List[float]
            a_intervals.append(a_confidence_interval)

            theta_i_interval = [np.arccos(1 - 2 * a_i) / 2 / np.pi  # type: ignore
                                for a_i in a_confidence_interval]
            theta_intervals.append(theta_i_interval)
            num_oracle_queries = 0  # no Q-oracle call, only a single one to A

        else:
            num_iterations = 0  # keep track of the number of iterations
            shots = self._quantum_instance._run_config.shots  # number of shots per iteration

            # do while loop, keep in mind that we scaled theta mod 2pi such that it lies in [0,1]
            while theta_intervals[-1][1] - theta_intervals[-1][0] > self._epsilon / np.pi:
                num_iterations += 1

                # get the next k
                k, upper_half_circle = self._find_next_k(powers[-1], upper_half_circle,
                                                         theta_intervals[-1],  # type: ignore
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
                one_counts, prob = self._probability_to_measure_one(counts)  # type: ignore
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
                if self._confint_method == 'chernoff':
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
                a_u = cast(float, a_u)
                a_l = cast(float, a_l)
                a_intervals.append([a_l, a_u])

        # get the latest confidence interval for the estimate of a
        a_confidence_interval = a_intervals[-1]

        # the final estimate is the mean of the confidence interval
        value = np.mean(a_confidence_interval)

        # transform to estimate
        estimation = self.post_processing(value)  # type: ignore
        confidence_interval = [self.post_processing(x) for x in a_confidence_interval]

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

        ae_result = AmplitudeEstimationAlgorithmResult()
        ae_result.value = self._ret['value']
        ae_result.estimation = self._ret['estimation']
        ae_result.num_oracle_queries = self._ret['num_oracle_queries']
        ae_result.confidence_interval = self._ret['confidence_interval']

        result = IterativeAmplitudeEstimationResult()
        result.combine(ae_result)
        result.value_confidence_interval = self._ret['value_confidence_interval']
        result.alpha = self._ret['alpha']
        result.actual_epsilon = self._ret['actual_epsilon']
        result.a_intervals = self._ret['a_intervals']
        result.theta_intervals = self._ret['theta_intervals']
        result.powers = self._ret['powers']
        result.ratios = self._ret['ratios']
        return result


class IterativeAmplitudeEstimationResult(AmplitudeEstimationAlgorithmResult):
    """ IterativeAmplitudeEstimation Result."""

    @property
    def value_confidence_interval(self) -> List[float]:
        """ return value_confidence_interval  """
        return self.get('value_confidence_interval')

    @value_confidence_interval.setter
    def value_confidence_interval(self, value: List[float]) -> None:
        """ set value_confidence_interval """
        self.data['value_confidence_interval'] = value

    @property
    def alpha(self) -> float:
        """ return alpha """
        return self.get('alpha')

    @alpha.setter
    def alpha(self, value: float) -> None:
        """ set alpha """
        self.data['alpha'] = value

    @property
    def actual_epsilon(self) -> float:
        """ return mle """
        return self.get('actual_epsilon')

    @actual_epsilon.setter
    def actual_epsilon(self, value: float) -> None:
        """ set mle """
        self.data['actual_epsilon'] = value

    @property
    def a_intervals(self) -> List[List[float]]:
        """ return a_intervals """
        return self.get('a_intervals')

    @a_intervals.setter
    def a_intervals(self, value: List[List[float]]) -> None:
        """ set a_intervals """
        self.data['a_intervals'] = value

    @property
    def theta_intervals(self) -> List[List[float]]:
        """ return theta_intervals """
        return self.get('theta_intervals')

    @theta_intervals.setter
    def theta_intervals(self, value: List[List[float]]) -> None:
        """ set theta_intervals """
        self.data['theta_intervals'] = value

    @property
    def powers(self) -> List[int]:
        """ return powers """
        return self.get('powers')

    @powers.setter
    def powers(self, value: List[int]) -> None:
        """ set powers """
        self.data['powers'] = value

    @property
    def ratios(self) -> List[float]:
        """ return ratios """
        return self.get('ratios')

    @ratios.setter
    def ratios(self, value: List[float]) -> None:
        """ set ratios """
        self.data['ratios'] = value

    @staticmethod
    def from_dict(a_dict: Dict) -> 'IterativeAmplitudeEstimationResult':
        """ create new object from a dictionary """
        return IterativeAmplitudeEstimationResult(a_dict)
