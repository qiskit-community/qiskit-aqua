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
import warnings
from typing import List

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.initial_states import InitialState

logger = logging.getLogger(__name__)


class VSCF(QuantumCircuit, InitialState):
    r"""Initial state for vibrational modes.

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
        # get the bitstring encoding initial state
        bitstr = _build_bitstr(basis)
        self._bitstr = bitstr

        # construct the circuit
        qr = QuantumRegister(len(bitstr), 'q')
        super().__init__(qr, name='VSCF')

        # add gates in the right positions
        for i, bit in enumerate(reversed(bitstr)):
            if bit:
                self.x(i)

    def construct_circuit(self, mode='circuit', register=None):
        """Construct the statevector of desired initial state.

        Args:
            mode (string): `vector` or `circuit`. The `vector` mode produces the vector.
                            While the `circuit` constructs the quantum circuit corresponding that
                            vector.
            register (QuantumRegister): register for circuit construction.

        Returns:
            QuantumCircuit or numpy.ndarray: statevector.

        Raises:
            ValueError: when mode is not 'vector' or 'circuit'.
        """
        warnings.warn('The VSCF.construct_circuit method is deprecated as of Aqua 0.9.0 and '
                      'will be removed no earlier than 3 months after the release. The HarteeFock '
                      'class is now a QuantumCircuit instance and can directly be used as such.',
                      DeprecationWarning, stacklevel=2)
        if mode == 'vector':
            state = 1.0
            one = np.asarray([0.0, 1.0])
            zero = np.asarray([1.0, 0.0])
            for k in self._bitstr[::-1]:
                state = np.kron(one if k else zero, state)
            return state
        elif mode == 'circuit':
            if register is None:
                register = QuantumRegister(self.num_qubits, name='q')
            quantum_circuit = QuantumCircuit(register)
            quantum_circuit.compose(self, inplace=True)
            return quantum_circuit
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')

    @property
    def bitstr(self):
        """Getter of the bit string represented the statevector."""
        warnings.warn('The VSCF.bitstr property is deprecated as of Aqua 0.9.0 and will be '
                      'removed no earlier than 3 months after the release. To get the bitstring '
                      'you can use the quantum_info.Statevector class and the probabilities_dict '
                      'method.', DeprecationWarning, stacklevel=2)
        return self._bitstr


def _build_bitstr(basis):
    num_qubits = sum(basis)
    bitstr = np.zeros(num_qubits, np.bool)
    count = 0
    for modal in basis:
        bitstr[num_qubits - count - 1] = True
        count += modal

    return bitstr
