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

"""The Grover operator."""

from typing import List, Optional
from qiskit.circuit import QuantumCircuit, QuantumRegister


class GroverOperator(QuantumCircuit):
    """The Grover operator."""

    def __init__(self, oracle: QuantumCircuit,
                 a_operator: Optional[QuantumCircuit] = None,
                 zero_reflection: Optional[QuantumCircuit] = None,
                 idle_qubits: Optional[List[int]] = None,
                 name: str = 'Q') -> None:
        """
        Args:
            oracle: The oracle implementing a reflection about the bad state.
            a_operator: The operator preparing the good and bad state. For Grover's algorithm,
                this is a n-qubit Hadamard gate and for Amplitude Amplification or Estimation
                the operator A.
            zero_reflection: The reflection about the zero state.
            idle_qubits: Qubits that are ignored in the reflection about zero.
            name: The name of the circuit.
        """
        super().__init__(name=name)
        self._oracle = oracle
        self._a_operator = a_operator
        self._zero_reflection = zero_reflection
        self._idle_qubits = idle_qubits

        self._build()

    @property
    def num_state_qubits(self):
        """The number of state qubits."""
        if hasattr(self._oracle, 'num_state_qubits'):
            return self._oracle.num_state_qubits
        return self._oracle.num_qubits

    @property
    def num_ancilla_qubits(self) -> int:
        """The number of ancilla qubits.

        Returns:
            The number of ancilla qubits in the circuit.
        """
        max_num_ancillas = 0
        if self._zero_reflection:
            max_num_ancillas = self._zero_reflection.num_ancilla_qubits
        elif self._oracle.num_qubits > 1:
            max_num_ancillas = 1

        if self._a_operator and hasattr(self._a_operator, 'num_ancilla_qubits'):
            max_num_ancillas = max(max_num_ancillas, self._a_operator.num_ancilla_qubits)

        if hasattr(self._a_operator, 'num_ancilla_qubits'):
            max_num_ancillas = max(max_num_ancillas, self._oracle.num_ancilla_qubits)

        return max_num_ancillas

    @property
    def num_qubits(self):
        """The number of qubits in the Grover operator."""
        return self.num_state_qubits + self.num_ancilla_qubits

    @property
    def zero_reflection(self) -> QuantumCircuit:
        """The subcircuit implementing the reflection about 0."""
        if self._zero_reflection is not None:
            return self._zero_reflection

        zero_reflection = QuantumCircuit(self.num_qubits, name='S_0')
        if self.num_qubits == 1:
            zero_reflection.z(0)
        else:
            zero_reflection.x(list(range(self.num_qubits)))
            zero_reflection.h(self.num_qubits - 1)
            zero_reflection.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
            zero_reflection.h(self.num_qubits - 1)
            zero_reflection.x(list(range(self.num_qubits)))

        return zero_reflection

    @property
    def a_operator(self) -> QuantumCircuit:
        """The subcircuit implementing the A operator or Hadamards."""
        if self._a_operator:
            return self._a_operator

        hadamards = QuantumCircuit(self.num_state_qubits, name='H')
        hadamards.h(list(range(self.num_state_qubits)))
        return hadamards

    @property
    def oracle(self):
        """The oracle implementing a relfection about the bad state."""
        return self._oracle

    def _build(self):
        self.qregs = [QuantumRegister(self.num_state_qubits, name='state')]
        if self.num_ancilla_qubits > 0:
            self.qregs += [QuantumRegister(self.num_ancilla_qubits, name='ancilla')]

        _append(self, self.oracle)
        _append(self, self.a_operator.inverse())
        _append(self, self.zero_reflection)
        _append(self, self.a_operator)


def _append(target, other, qubits=None, ancillas=None):
    if hasattr(other, 'num_state_qubits') and hasattr(other, 'num_ancilla_qubits'):
        num_state_qubits = other.num_state_qubits
        num_ancilla_qubits = other.num_ancilla_qubits
    else:
        num_state_qubits = other.num_qubits
        num_ancilla_qubits = 0

    if qubits is None:
        qubits = list(range(num_state_qubits))
    elif isinstance(qubits, QuantumRegister):
        qubits = qubits[:]

    if num_ancilla_qubits > 0:
        if ancillas is None:
            qubits += list(range(num_state_qubits, num_state_qubits + num_ancilla_qubits))
        else:
            qubits += ancillas[:num_ancilla_qubits]

    target.append(other.to_instruction(), qubits)
