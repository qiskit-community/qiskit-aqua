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

from qiskit.chemistry.circuit.library.initial_states.vscf import vscf_bitstring

logger = logging.getLogger(__name__)


class VSCF(InitialState):
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
        bitstr = vscf_bitstring(basis)
        self._bitstr = bitstr

        super().__init__()

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
        if mode == 'vector':
            state = 1.0
            one = np.asarray([0.0, 1.0])
            zero = np.asarray([1.0, 0.0])
            for k in self._bitstr[::-1]:
                state = np.kron(one if k else zero, state)
            return state
        elif mode == 'circuit':
            if register is None:
                register = QuantumRegister(len(self.bitstr), name='q')
            quantum_circuit = QuantumCircuit(register, name='VSCF')
            for i, bit in enumerate(reversed(self.bitstr)):
                if bit:
                    quantum_circuit.x(i)

            return quantum_circuit
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')

    @property
    def bitstr(self):
        """Getter of the bit string represented the statevector."""
        return self._bitstr
