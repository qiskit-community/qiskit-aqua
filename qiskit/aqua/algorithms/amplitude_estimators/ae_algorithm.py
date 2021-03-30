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

"""The Amplitude Estimation Algorithm."""

from typing import Optional, Union, List, Callable, Dict
import logging
import warnings
from abc import abstractmethod

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QuantumAlgorithm, AlgorithmResult
from qiskit.aqua.utils import CircuitFactory
from .q_factory import QFactory
from ...deprecation import warn_package

logger = logging.getLogger(__name__)


class AmplitudeEstimationAlgorithm(QuantumAlgorithm):
    r"""The Quantum Amplitude Estimation (QAE) algorithm base class.

    In general, QAE algorithms aim to approximate the amplitude of a certain, marked state.
    This amplitude is encoded in the so-called A operator, performing the mapping

    .. math::

            \mathcal{A}|0\rangle_n = \sqrt{1 - a} |\Psi_0\rangle_n + \sqrt{a} |\Psi_1\rangle_n

    where the amplitude `a` (in [0, 1]) is approximated, and :math:`|\Psi_0\rangle` and
    :math:`|\Psi_1\rangle` are two orthonormal states.
    In the QAE algorithms, the Grover operator :math:`\mathcal{Q}` is used, which is defined as

    .. math::

            \mathcal{Q} = -\mathcal{A} \mathcal{S}_0 \mathcal{A}^{-1} \mathcal{S}_{\Psi_0},

    where :math:`\mathcal{S}_0` reflects about the :math:`|0\rangle_n` state and
    :math:`\mathcal{S}_{\Psi_0}` reflects about :math:`|\Psi_0\rangle_n`.

    See [1] for more detail about QAE.

    References:
        [1]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
             Quantum Amplitude Amplification and Estimation.
             `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.

    """

    @abstractmethod
    def __init__(self,
                 state_preparation: Optional[Union[QuantumCircuit, CircuitFactory]] = None,
                 grover_operator: Optional[Union[QuantumCircuit, CircuitFactory]] = None,
                 objective_qubits: Optional[List[int]] = None,
                 post_processing: Optional[Callable[[float], float]] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
                 a_factory: Optional[CircuitFactory] = None,
                 q_factory: Optional[CircuitFactory] = None,
                 i_objective: Optional[int] = None) -> None:
        r"""
        Args:
            state_preparation: The :math:`\mathcal{A}` operator, specifying the QAE problem.
            grover_operator: The :math:`\mathcal{Q}` operator (Grover operator), constructed from
                the :math:`\mathcal{A}` operator.
            objective_qubits: A list of qubit indices. A measurement outcome is classified as
                'good' state if all objective qubits are in state :math:`|1\rangle`, otherwise it
                is classified as 'bad'.
            post_processing: A mapping applied to the estimate of :math:`0 \leq a \leq 1`,
                usually used to map the estimate to a target interval.
            quantum_instance: The backend (or `QuantumInstance`) to execute the circuits on.
            a_factory: Deprecated, use ``state_preparation``. The A operator, specifying the QAE
                problem.
            q_factory: Deprecated, use ``grover_operator``.
                The Q operator (Grover operator), constructed from the A operator.
            i_objective: Deprecated use ``objective_qubits``.
                Index of the objective qubit, that marks the 'good/bad' states
        """
        warn_package('aqua.algorithms.amplitude_estimators',
                     'qiskit.algorithms.amplitude_estimators', 'qiskit-terra')
        if isinstance(state_preparation, CircuitFactory) or a_factory is not None:
            warnings.warn('Passing a CircuitFactory as A operator is deprecated as of 0.8.0, '
                          'this feature will be removed no earlier than 3 months after the '
                          'release. You should pass a QuantumCircuit instead.',
                          DeprecationWarning, stacklevel=2)
            if isinstance(state_preparation, CircuitFactory):
                a_factory = state_preparation
                state_preparation = None

        if isinstance(grover_operator, CircuitFactory) or q_factory is not None:
            warnings.warn('Passing a CircuitFactory as Q operator is deprecated as of 0.8.0, '
                          'this feature will be removed no earlier than 3 months after the '
                          'release. You should pass a QuantumCircuit instead.',
                          DeprecationWarning, stacklevel=2)
            if isinstance(grover_operator, CircuitFactory):
                q_factory = grover_operator
                grover_operator = None

        if i_objective is not None:
            warnings.warn('The i_objective argument is deprecated as of 0.8.0 and will be removed '
                          'no earlier than 3 months after the release. You should use the '
                          'objective_qubits argument instead.', DeprecationWarning, stacklevel=2)

        self._a_factory = a_factory
        self._q_factory = q_factory
        self._i_objective = i_objective
        self._objective_qubits = objective_qubits
        self._state_preparation = state_preparation
        self._grover_operator = grover_operator
        self._post_processing = (lambda x: x) if post_processing is None else post_processing

        super().__init__(quantum_instance)

    @property
    def state_preparation(self) -> QuantumCircuit:
        r"""Get the :math:`\mathcal{A}` operator encoding the amplitude :math:`a`.

        Returns:
            The :math:`\mathcal{A}` operator as `QuantumCircuit`.
        """
        return self._state_preparation

    @state_preparation.setter
    def state_preparation(self, state_preparation: QuantumCircuit) -> None:
        r"""Set the :math:`\mathcal{A}` operator, that encodes the amplitude to be estimated.

        Args:
            state_preparation: The new :math:`\mathcal{A}` operator.
        """
        self._state_preparation = state_preparation

    @property
    def grover_operator(self) -> Optional[QuantumCircuit]:
        r"""Get the :math:`\mathcal{Q}` operator, or Grover operator.

        If the Grover operator is not set, we try to build it from the :math:`\mathcal{A}` operator
        and `objective_qubits`. This only works if `objective_qubits` is a list of integers.

        Returns:
            The Grover operator, or None if neither the Grover operator nor the
            :math:`\mathcal{A}` operator is  set.
        """
        if self._grover_operator is not None:
            return self._grover_operator

        if self.state_preparation is not None and isinstance(self.objective_qubits, list):
            # build the reflection about the bad state
            num_state_qubits = self.state_preparation.num_qubits \
                - self.state_preparation.num_ancillas

            oracle = QuantumCircuit(num_state_qubits)
            oracle.h(self.objective_qubits[-1])
            if len(self.objective_qubits) == 1:
                oracle.x(self.objective_qubits[0])
            else:
                oracle.mcx(self.objective_qubits[:-1], self.objective_qubits[-1])
            oracle.h(self.objective_qubits[-1])

            # construct the grover operator
            return GroverOperator(oracle, self.state_preparation)

        return None

    @grover_operator.setter
    def grover_operator(self, grover_operator: QuantumCircuit) -> None:
        r"""Set the :math:`\mathcal{Q}` operator.

        Args:
            grover_operator: The new :math:`\mathcal{Q}` operator.
        """
        self._grover_operator = grover_operator

    @property
    def objective_qubits(self) -> Optional[List[int]]:
        """Get the criterion for a measurement outcome to be in a 'good' state.

        Returns:
            The criterion as list of qubit indices.
        """
        if self._objective_qubits is not None:
            return self._objective_qubits

        # by default the last qubit of the input state is the objective qubit
        if self._state_preparation is not None:
            return [self._state_preparation.num_qubits - 1]

        # check the deprecated locations (cannot use property since this emits a warning)
        if self._i_objective is not None:
            return [self._i_objective]

        if self._q_factory is not None:
            return [self._q_factory.i_objective]

        if self._a_factory is not None:
            return [self._a_factory.num_target_qubits - 1]

        return None

    @objective_qubits.setter
    def objective_qubits(self, objective_qubits: List[int]):
        """Set the criterion for a measurement outcome to be in a 'good' state.

        Args:
            objective_qubits: The criterion as callable of list of qubit indices.
        """
        self._objective_qubits = objective_qubits

    def is_good_state(self, measurement: str) -> bool:
        """Determine whether a given state is a good state.

        Args:
            measurement: A measurement as bitstring, e.g. '01100'.

        Returns:
            True if the measurement corresponds to a good state, False otherwise.

        Raises:
            ValueError: If ``self.objective_qubits`` is not set.
        """
        if self.objective_qubits is None:
            raise ValueError('is_good_state can only be called if objective_qubits is set.')

        return all(measurement[objective] == '1' for objective in self.objective_qubits)

    @property
    def a_factory(self):
        r"""Get the A operator encoding the amplitude `a` that's approximated, i.e.

            A \|0>_n \|0> = sqrt{1 - a} \|psi_0>_n \|0> + sqrt{a} \|psi_1>_n \|1>

        see the original Brassard paper (https://arxiv.org/abs/quant-ph/0005055) for more detail.

        Returns:
            CircuitFactory: the A operator as CircuitFactory
        """
        warnings.warn('The a_factory property is deprecated as of 0.8.0 and will be removed no '
                      'earlier than 3 months after the release. You should use the '
                      'state_preparation property instead.', DeprecationWarning, stacklevel=2)
        return self._a_factory

    @a_factory.setter
    def a_factory(self, a_factory):
        """Set the A operator, that encodes the amplitude to be estimated.

        Args:
            a_factory (CircuitFactory): the A Operator
        """
        warnings.warn('The a_factory setter is deprecated as of 0.8.0 and will be removed no '
                      'earlier than 3 months after the release. You should use the '
                      'state_preparation setter instead, which takes a QuantumCircuit instead of '
                      'a CircuitFactory.', DeprecationWarning, stacklevel=2)
        self._a_factory = a_factory

    @property
    def q_factory(self):
        r"""Get the Q operator, or Grover-operator for the Amplitude Estimation algorithm, i.e.

        .. math::

            \mathcal{Q} = \mathcal{A} \mathcal{S}_0 \mathcal{A}^\dagger \mathcal{S}_f,

        where :math:`\mathcal{S}_0` reflects about the \|0>_n state and S_psi0 reflects about
        :math:`|\Psi_0\rangle_n`.
        See https://arxiv.org/abs/quant-ph/0005055 for more detail.

        If the Q operator is not set, we try to build it from the A operator.
        If neither the A operator is set, None is returned.

        Returns:
            QFactory: returns the current Q factory of the algorithm
        """
        warnings.warn('The q_factory property is deprecated as of 0.8.0 and will be removed no '
                      'earlier than 3 months after the release. You should use the grover_operator '
                      'property instead.', DeprecationWarning, stacklevel=2)
        if self._q_factory is not None:
            return self._q_factory

        if self._a_factory is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                q_factory = QFactory(self._a_factory, self.i_objective)
            return q_factory

        return None

    @q_factory.setter
    def q_factory(self, q_factory):
        """
        Set the Q operator as QFactory.

        Args:
            q_factory (QFactory): the specialized Q operator
        """
        warnings.warn('The q_factory setter is deprecated as of 0.8.0 and will be removed no '
                      'earlier than 3 months after the release. You should use the grover_operator '
                      'setter instead, which takes a QuantumCircuit instead of a CircuitFactory.',
                      DeprecationWarning, stacklevel=2)
        self._q_factory = q_factory

    @property
    def i_objective(self):
        r"""
        Get the index of the objective qubit. The objective qubit marks the \|psi_0> state (called
        'bad states' in https://arxiv.org/abs/quant-ph/0005055)
        with \|0> and \|psi_1> ('good' states) with \|1>.
        If the A operator performs the mapping

            A \|0>_n \|0> = sqrt{1 - a} \|psi_0>_n \|0> + sqrt{a} \|psi_1>_n \|1>

        then, the objective qubit is the last one (which is either \|0> or \|1>).

        If the objective qubit (i_objective) is not set, we check if the Q operator (q_factory) is
        set and return the index specified there. If the q_factory is not defined,
        the index equals the number of qubits of the A operator (a_factory) minus one.
        If also the a_factory is not set, return None.

        Returns:
            int: the index of the objective qubit
        """
        warnings.warn('The i_objective property is deprecated as of 0.8.0 and will be removed no '
                      'earlier than 3 months after the release. You should use the '
                      'objective_qubits property instead.', DeprecationWarning, stacklevel=2)

        if self._i_objective is not None:
            return self._i_objective

        if self._q_factory is not None:
            return self._q_factory.i_objective

        if self._a_factory is not None:
            return self._a_factory.num_target_qubits - 1

        return None

    @i_objective.setter
    def i_objective(self, i_objective):
        """
        Set the index of the objective qubit, i.e. the qubit deciding between 'good/bad' states.

        Args:
            i_objective (int): the index

        Note:
            No checks about the validity of the index are performed, since i_objective could also
            be set before the A/Q operators and in that case checks cannot be done.
        """
        warnings.warn('The i_objective setter is deprecated as of 0.8.0 and will be removed no '
                      'earlier than 3 months after the release. You should use the '
                      'objective_qubits setter instead, which takes a List[int] instead of an int.',
                      DeprecationWarning, stacklevel=2)
        self._i_objective = i_objective

    def post_processing(self, value: float) -> float:
        r"""Post processing of the raw amplitude estimation output :math:`0 \leq a \leq 1`.

        Args:
            value: The estimation value :math:`a`.

        Returns:
            The value after post processing, usually mapping the interval :math:`[0, 1]`
            to the target interval.
        """
        if self._a_factory is not None:
            return self._a_factory.value_to_estimation(value)
        return self._post_processing(value)


class AmplitudeEstimationAlgorithmResult(AlgorithmResult):
    """ AmplitudeEstimationAlgorithm Result."""

    @property
    def a_estimation(self) -> float:
        """ return a_estimation """
        return self.get('a_estimation')

    @a_estimation.setter
    def a_estimation(self, value: float) -> None:
        """ set a_estimation """
        self.data['a_estimation'] = value

    @property
    def estimation(self) -> float:
        """ return estimation """
        return self.get('estimation')

    @estimation.setter
    def estimation(self, value: float) -> None:
        """ set estimation """
        self.data['estimation'] = value

    @property
    def num_oracle_queries(self) -> int:
        """ return num_oracle_queries """
        return self.get('num_oracle_queries')

    @num_oracle_queries.setter
    def num_oracle_queries(self, value: int) -> None:
        """ set num_oracle_queries """
        self.data['num_oracle_queries'] = value

    @property
    def confidence_interval(self) -> List[float]:
        """ return confidence_interval """
        return self.get('confidence_interval')

    @confidence_interval.setter
    def confidence_interval(self, value: List[float]) -> None:
        """ set confidence_interval """
        self.data['confidence_interval'] = value

    @staticmethod
    def from_dict(a_dict: Dict) -> 'AmplitudeEstimationAlgorithmResult':
        """ create new object from a dictionary """
        return AmplitudeEstimationAlgorithmResult(a_dict)

    def __getitem__(self, key: object) -> object:
        if key == '95%_confidence_interval':
            warnings.warn('95%_confidence_interval deprecated, use confidence_interval property.',
                          DeprecationWarning)
            return super().__getitem__('confidence_interval')
        elif key == 'value':
            warnings.warn('value deprecated, use a_estimation property.', DeprecationWarning)
            return super().__getitem__('a_estimation')

        return super().__getitem__(key)
