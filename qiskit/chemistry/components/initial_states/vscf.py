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

"""Initial state for vibrational modes."""

from typing import List
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit


class VSCF(QuantumCircuit):
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

        # get the bitstring encoding initial state
        bitstr = _build_bitstr(basis)

        # construct the circuit
        qr = QuantumRegister(len(bitstr), 'q')
        super().__init__(qr, name='VSCF')

        # add gates in the right positions
        for i, bit in enumerate(reversed(bitstr)):
            if bit:
                self.x(i)


def _build_bitstr(basis):
    num_qubits = sum(basis)
    bitstr = np.zeros(num_qubits, np.bool)
    count = 0
    for modal in basis:
        bitstr[num_qubits - count - 1] = True
        count += modal

    return bitstr
