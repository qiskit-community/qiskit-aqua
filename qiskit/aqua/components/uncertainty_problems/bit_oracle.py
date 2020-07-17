# -*- coding: utf-8 -*-

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

"""The Bit oracle."""

from typing import List
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import MCXGate


class BitOracle(QuantumCircuit):
    """The bit oracle.

    Adds a -1 phase if all objective qubits are in state 1.
    """

    def __init__(self, num_state_qubits: int,
                 objective_qubits: List[int],
                 mcx: str = 'noancilla',
                 name: str = 'S_f') -> None:
        """
        Args:
            num_state_qubits: The number of qubits.
            objective_qubits: The objective qubits.
            mcx: The mode for the multi-controlled X gate.
            name: The name of the circuit.

        Raises:
            ValueError: If ``objective_qubits`` contains an invalid index.
        """
        qr_state = QuantumRegister(num_state_qubits, 'state')
        super().__init__(qr_state, name=name)

        num_ancillas = MCXGate.get_num_ancilla_qubits(len(objective_qubits) - 1, mcx)
        if num_ancillas > 0:
            qr_ancilla = AncillaRegister(num_ancillas, 'ancilla')
            self.add_register(qr_ancilla)
        else:
            qr_ancilla = []

        if any(qubit >= num_state_qubits for qubit in objective_qubits):
            raise ValueError('Qubit index out of range, max {}, provided {}'.format(
                num_state_qubits, objective_qubits))

        if num_state_qubits == 1:
            self.x(0)
            self.z(0)
            self.x(0)
        else:
            self.x(objective_qubits)
            self.h(objective_qubits[-1])
            if len(objective_qubits) == 1:
                self.x(objective_qubits[0])
            else:
                self.mcx(objective_qubits[:-1], objective_qubits[-1], qr_ancilla[:], mode=mcx)
            self.h(objective_qubits[-1])
            self.x(objective_qubits)
