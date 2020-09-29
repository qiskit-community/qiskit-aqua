# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Initial state for vibrational modes. """

import logging
from typing import List

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua.components.initial_states import InitialState

logger = logging.getLogger(__name__)


class VSCF(InitialState):
    r""" Initial state for vibrational modes.

    Creates an occupation number vector as defined in
    Ollitrault Pauline J., Chemical science 11 (2020): 6842-6855.
    e.g. for 2 modes with 4 modals per mode it creates: \|1000 1000>
    """

    def __init__(self, basis: List[int]) -> None:
        """
        Args:
            basis: Is a list defining the number of modals per mode. E.g. for a 3 modes system
                with 4 modals per mode basis = [4,4,4]
        """
        super().__init__()
        self._basis = basis
        self._num_qubits = sum(basis)

        self._bitstr = self._build_bitstr()

    def _build_bitstr(self) -> np.ndarray:

        bitstr = np.zeros(self._num_qubits, np.bool)
        count = 0
        for i in range(len(self._basis)):
            bitstr[self._num_qubits-count-1] = True
            count += self._basis[i]

        return bitstr

    def construct_circuit(self, mode: str = 'circuit', register: QuantumRegister = None) \
            -> QuantumCircuit:
        """
        Construct the circuit of desired initial state.

        Args:
            mode: `vector` or `circuit`. The `vector` mode produces the vector.
                While the `circuit` constructs the quantum circuit corresponding that vector.
            register (QuantumRegister): register for circuit construction.

        Returns:
            QuantumCircuit or numpy.ndarray: statevector.

        Raises:
            ValueError: when mode is not 'vector' or 'circuit'.
        """
        if self._bitstr is None:
            self._build_bitstr()
        if mode == 'vector':
            state = 1.0
            one = np.asarray([0.0, 1.0])
            zero = np.asarray([1.0, 0.0])
            for k in self._bitstr[::-1]:
                state = np.kron(one if k else zero, state)
            return state
        elif mode == 'circuit':
            if register is None:
                register = QuantumRegister(self._num_qubits, name='q')
            quantum_circuit = QuantumCircuit(register)
            for qubit_idx, bit in enumerate(self._bitstr[::-1]):
                if bit:
                    quantum_circuit.u(np.pi, 0.0, np.pi, register[qubit_idx])
            return quantum_circuit
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')

    @property
    def bitstr(self) -> np.ndarray:
        """Getter of the bit string represented the statevector.

        Returns:
             numpy.ndarray containing the bitstring representation
        """
        if self._bitstr is None:
            self._build_bitstr()
        return self._bitstr
