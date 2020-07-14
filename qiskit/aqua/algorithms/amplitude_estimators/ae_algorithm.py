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

"""The Amplitude Estimation Algorithm."""

from typing import Optional, Union, List, Callable
import logging
import warnings
from abc import abstractmethod

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.utils import CircuitFactory
from .q_factory import QFactory

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
        `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_

    """

    @abstractmethod
    def __init__(self,
                 state_in: Optional[Union[QuantumCircuit, CircuitFactory]] = None,
                 grover_operator: Optional[Union[QuantumCircuit, CircuitFactory]] = None,
                 is_good_state: Optional[Union[callable, List[int]]] = None,
                 post_processing: Optional[Callable[[float], float]] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None,
                 a_factory: Optional[CircuitFactory] = None,
                 q_factory: Optional[CircuitFactory] = None,
                 i_objective: Optional[int] = None) -> None:
        r"""
        Args:
            state_in: The :math:`\mathcal{A}` operator, specifying the QAE problem.
            grover_operator: The :math:`\mathcal{Q}` operator (Grover operator), constructed from
                the :math:`\mathcal{A}` operator.
            is_good_state: A function to determine if a measurement is part of the 'good' state
                or 'bad' state. If a list of integers indices is passed, a state is marked as good
                if the qubits at these indices are :math:`|1\rangle`.
            quantum_instance: The backend (or `QuantumInstance`) to execute the circuits on.
            a_factory: Deprecated, use ``state_in``. The A operator, specifying the QAE problem.
            q_factory: Deprecated, use ``grover_operator``.
                The Q operator (Grover operator), constructed from the A operator.
            i_objective: Deprecated use ``is_good_state``.
                Index of the objective qubit, that marks the 'good/bad' states
        """
        # self._a_factory = state_in
        # self._q_factory = grover_operator
        self._a_factory = None
        self._q_factory = None
        self._i_objective = None

        if isinstance(state_in, CircuitFactory) or a_factory is not None:
            warnings.warn('Passing a CircuitFactory as A operator is deprecated as of 0.8.0, '
                          'this feature will be removed no earlier than 3 months after the release.'
                          'You should pass a QuantumCircuit instead.',
                          DeprecationWarning, stacklevel=2)
            if a_factory is not None:
                self._a_factory = a_factory

        if isinstance(grover_operator, CircuitFactory) or q_factory is not None:
            warnings.warn('Passing a CircuitFactory as Q operator is deprecated as of 0.8.0, '
                          'this feature will be removed no earlier than 3 months after the release.'
                          'You should pass a QuantumCircuit instead.',
                          DeprecationWarning, stacklevel=2)
            if q_factory is not None:
                self._q_factory = q_factory

        if i_objective is not None:
            warnings.warn('The i_objective argument is deprecated as of 0.8.0 and will be removed '
                          'no earlier than 3 months after the release. You should use the '
                          'is_good_state argument instead.', DeprecationWarning, stacklevel=2)
            self._i_objective = i_objective
            self._is_good_state = [i_objective]
        else:
            self._is_good_state = is_good_state

        self._state_in = state_in
        self._grover_operator = grover_operator
        self._post_processing = (lambda x: x) if post_processing is None else post_processing

        super().__init__(quantum_instance)

    @property
    def state_in(self) -> QuantumCircuit:
        r"""Get the :math:`\mathcal{A}` operator encoding the amplitude :math:`a`.

        Returns:
            The :math:`\mathcal{A}` operator as `QuantumCircuit`.
        """
        return self._state_in

    @state_in.setter
    def state_in(self, state_in: QuantumCircuit) -> None:
        r"""Set the :math:`\mathcal{A}` operator, that encodes the amplitude to be estimated.

        Args:
            state_in: The new :math:`\mathcal{A}` operator.
        """
        self._state_in = state_in

    @property
    def grover_operator(self) -> Optional[QuantumCircuit]:
        r"""Get the :math:`\mathcal{Q}` operator, or Grover operator.

        If the Grover operator is not set, we try to build it from the :math:`\mathcal{A}` operator
        and `is_good_state`. This only works if `is_good_state` is a list of integers.

        Returns:
            The Grover operator, or None if neither the Grover operator nor the
            :math:`\mathcal{A}` operator is  set.
        """
        if self._grover_operator is not None:
            return self._grover_operator

        if self._state_in is not None and isinstance(self._is_good_state, list):
            from qiskit.aqua.components.uncertainty_problems.bit_oracle import BitOracle
            from qiskit.aqua.components.uncertainty_problems.grover_operator import GroverOperator

            # build the reflection about the bad state
            num_state_qubits = self._state_in.num_qubits - self._state_in.num_ancillas
            oracle = BitOracle(num_state_qubits, objective_qubits=self._is_good_state)

            # construct the grover operator
            return GroverOperator(oracle, self._state_in)

        return None

    @grover_operator.setter
    def grover_operator(self, grover_operator: QuantumCircuit) -> None:
        r"""Set the :math:`\mathcal{Q}` operator.

        Args:
            grover_operator: The new :math:`\mathcal{Q}` operator.
        """
        self._grover_operator = grover_operator

    @property
    def is_good_state(self) -> Union[callable, List[int]]:
        """Get the criterion for a measurement outcome to be in a 'good' state.

        Returns:
            The criterion as callable of list of qubit indices.
        """
        return self._is_good_state

    @is_good_state.setter
    def is_good_state(self, is_good_state: Union[callable, List[int]]):
        """Set the criterion for a measurement outcome to be in a 'good' state.

        Args:
            is_good_state: The criterion as callable of list of qubit indices.
        """
        self._is_good_state = is_good_state

    @property
    def a_factory(self):
        r"""Get the A operator encoding the amplitude `a` that's approximated, i.e.

            A \|0>_n \|0> = sqrt{1 - a} \|psi_0>_n \|0> + sqrt{a} \|psi_1>_n \|1>

        see the original Brassard paper (https://arxiv.org/abs/quant-ph/0005055) for more detail.

        Returns:
            CircuitFactory: the A operator as CircuitFactory
        """
        warnings.warn('The a_factory property is deprecated as of 0.8.0 and will be removed no '
                      'earlier than 3 months after the release. You should use the state_in '
                      'property instead.', DeprecationWarning, stacklevel=2)
        return self._a_factory

    @a_factory.setter
    def a_factory(self, a_factory):
        """
        Set the A operator, that encodes the amplitude to be estimated.

        Args:
            a_factory (CircuitFactory): the A Operator
        """
        warnings.warn('The a_factory setter is deprecated as of 0.8.0 and will be removed no '
                      'earlier than 3 months after the release. You should use the state_in '
                      'setter instead, which takes a QuantumCircuit instead of a CircuitFactory.',
                      DeprecationWarning, stacklevel=2)
        self._a_factory = a_factory

    @property
    def q_factory(self):
        r"""
        Get the Q operator, or Grover-operator for the Amplitude Estimation algorithm, i.e.

            Q = -A S_0 A^{-1} S_psi0,

        where S_0 reflects about the \|0>_n state and S_psi0 reflects about \|psi_0>_n.
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
            return QFactory(self._a_factory, self.i_objective)

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
                      'earlier than 3 months after the release. You should use the is_good_state '
                      'property instead.', DeprecationWarning, stacklevel=2)

        if self._i_objective is not None:
            return self._i_objective

        if self._q_factory is not None and hasattr(self._q_factory, 'i_objective'):
            return self._q_factory.i_objective

        if self._a_factory is not None:
            if isinstance(self._a_factory, CircuitFactory):
                return self._a_factory.num_target_qubits - 1
            else:
                return (self._a_factory.num_qubits - self._a_factory.num_ancillas) - 1

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
                      'earlier than 3 months after the release. You should use the is_good_state '
                      'setter instead, which takes a List[int] instead of an int.',
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
        # if self.a_factory is not None:
        #     return self.a_factory.value_to_estimation(value)
        return self._post_processing(value)
