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
from qiskit.circuit import QuantumCircuit
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
        self._num_ancilla_qubits = MCXGate.get_num_ancilla_qubits(len(objective_qubits) - 1, mcx)
        super().__init__(num_state_qubits + self._num_ancilla_qubits, name=name)

        if any(qubit >= num_state_qubits for qubit in objective_qubits):
            raise ValueError('Qubit index out of range, max {}, provided {}'.format(
                num_state_qubits, objective_qubits
                )
            )

        ancilla_qubits = list(range(num_state_qubits, num_state_qubits + self._num_ancilla_qubits))

        if num_state_qubits == 1:
            self.x(0)
            self.z(0)
            self.x(0)
        else:
            self.x(objective_qubits)
            self.h(objective_qubits[-1])
            self.mcx(objective_qubits[:-1], objective_qubits[-1], ancilla_qubits, mode=mcx)
            self.h(objective_qubits[-1])
            self.x(objective_qubits)

    @property
    def num_ancilla_qubits(self) -> int:
        """The number of ancilla qubits."""
        return self._num_ancilla_qubits
