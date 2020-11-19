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

"""The custom initial state."""

from typing import Optional, Union
import logging
import numpy as np

from qiskit.circuit import QuantumRegister, QuantumCircuit, Qubit
from qiskit import execute as q_execute

from qiskit.aqua import AquaError, aqua_globals
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.circuits import StateVectorCircuit
from qiskit.aqua.utils.arithmetic import normalize_vector
from qiskit.aqua.utils.circuit_utils import convert_to_basis_gates
from qiskit.aqua.utils.validation import validate_in_set, validate_min
from qiskit.aqua.operators import StateFn

logger = logging.getLogger(__name__)


class Custom(InitialState):
    """
    The custom initial state.

    A custom initial state can be created with this component. It allows a state to be defined
    in the form of custom probability distribution with the *state_vector*, or by providing a
    desired *circuit* to set the state.

    Also *state* can be used having a few pre-defined initial states for convenience:

    - 'zero': configures the state vector with the zero probability distribution, and is
      effectively equivalent to the :class:`Zero` initial state.

    - 'uniform': This setting configures the state vector with the uniform probability distribution.
      All the qubits are set in superposition, each of them being initialized with a Hadamard gate,
      which means that a measurement will have equal probabilities to become :math:`1` or :math:`0`.

    - 'random': This setting assigns the elements of the state vector according to a random
      probability distribution.

    The custom initial state will be set from the *circuit*, the *state_vector*, or
    *state*, in that order. For *state_vector* the provided custom probability distribution
    will be internally normalized so the total probability represented is :math:`1.0`.

    """

    def __init__(self,
                 num_qubits: int,
                 state: str = 'zero',
                 state_vector: Optional[Union[np.ndarray, StateFn]] = None,
                 circuit: Optional[QuantumCircuit] = None) -> None:
        """
        Args:
            num_qubits: Number of qubits, has a minimum value of 1.
            state: Use a predefined state of ('zero' | 'uniform' | 'random')
            state_vector: An optional vector of ``complex`` or ``float`` representing the state as
                a probability distribution which will be normalized to a total probability of 1
                when initializing the qubits. The length of the vector must be :math:`2^q`, where
                :math:`q` is the *num_qubits* value. When provided takes precedence over *state*.
            circuit: A quantum circuit for the desired initial state. When provided takes
                precedence over both *state_vector* and *state*.
        Raises:
            AquaError: invalid input
        """
        validate_min('num_qubits', num_qubits, 1)
        validate_in_set('state', state, {'zero', 'uniform', 'random'})
        super().__init__()
        self._num_qubits = num_qubits
        self._state = state
        size = np.power(2, self._num_qubits)
        self._circuit = None
        if isinstance(state_vector, StateFn):
            state_vector = state_vector.to_matrix()
        # pylint: disable=comparison-with-callable
        if circuit is not None:
            if circuit.width() != num_qubits:
                logger.warning('The specified num_qubits and '
                               'the provided custom circuit do not match.')
            self._circuit = convert_to_basis_gates(circuit)
            if state_vector is not None:
                self._state = None
                self._state_vector = None
                logger.warning('The provided state_vector is ignored in favor of '
                               'the provided custom circuit.')
        else:
            if state_vector is None:
                if self._state == 'zero':
                    self._state_vector = np.array([1.0] + [0.0] * (size - 1))
                elif self._state == 'uniform':
                    self._state_vector = np.array([1.0 / np.sqrt(size)] * size)
                elif self._state == 'random':
                    self._state_vector = normalize_vector(aqua_globals.random.random(size))
                else:
                    raise AquaError('Unknown state {}'.format(self._state))
            else:
                if len(state_vector) != np.power(2, self._num_qubits):
                    raise AquaError('The state vector length {} is incompatible with '
                                    'the number of qubits {}'.format(
                                        len(state_vector), self._num_qubits))
                self._state_vector = normalize_vector(state_vector)
                self._state = None

    @staticmethod
    def _replacement():
        return 'Custom(state_vector=vector) is the same as a circuit where the ' \
                + '``initialize(vector/np.linalg.norm(vector))`` method has been called.'

    def construct_circuit(self, mode='circuit', register=None):
        # pylint: disable=import-outside-toplevel
        from qiskit import BasicAer

        if mode == 'vector':
            if self._state_vector is None:
                if self._circuit is not None:
                    self._state_vector = np.asarray(q_execute(self._circuit, BasicAer.get_backend(
                        'statevector_simulator')).result().get_statevector(self._circuit))
            return self._state_vector
        elif mode == 'circuit':
            if self._circuit is None:
                # create empty quantum circuit
                circuit = QuantumCircuit()

                if register is None:
                    register = QuantumRegister(self._num_qubits, name='q')

                if isinstance(register, QuantumRegister):
                    circuit.add_register(register)
                elif isinstance(register, list):
                    for q in register:
                        if isinstance(q, Qubit):
                            if not circuit.has_register(q.register):
                                circuit.add_register(q.register)
                        else:
                            raise AquaError('Unexpected qubit type {}.'.format(type(q)))
                else:
                    raise AquaError('Unexpected register type {}.'.format(type(register)))

                if self._state is None or self._state == 'random':
                    svc = StateVectorCircuit(self._state_vector)
                    svc.construct_circuit(circuit=circuit, register=register)
                elif self._state == 'uniform':
                    for i in range(self._num_qubits):
                        circuit.h(register[i])
                elif self._state == 'zero':
                    pass
                else:
                    AquaError('Unexpected state mode {}.'.format(self._state))
                self._circuit = circuit
            return self._circuit.copy()
        else:
            raise AquaError('Mode should be either "vector" or "circuit"')
