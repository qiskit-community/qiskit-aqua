# -*- coding: utf-8 -*-

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

"""
Abstract CircuitFactory to build a circuit, along with inverse, controlled
and power combinations of the circuit.
"""

from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.aqua.utils.controlled_circuit import get_controlled_circuit


class CircuitFactory(ABC):

    """ Base class for CircuitFactories """

    def __init__(self, num_target_qubits: int) -> None:
        self._num_target_qubits = num_target_qubits
        pass

    @property
    def num_target_qubits(self):
        """ Returns the number of target qubits """
        return self._num_target_qubits

    def required_ancillas(self):
        """ returns required ancillas """
        return 0

    def required_ancillas_controlled(self):
        """ returns required ancillas controlled """
        return self.required_ancillas()

    def get_num_qubits(self):
        """ returns number of qubits """
        return self._num_target_qubits + self.required_ancillas()

    def get_num_qubits_controlled(self):
        """ returns number of qubits controlled """
        return self._num_target_qubits + self.required_ancillas_controlled()

    @abstractmethod
    def build(self, qc, q, q_ancillas=None, params=None):
        """ Adds corresponding sub-circuit to given circuit

        Args:
            qc (QuantumCircuit): quantum circuit
            q (list): list of qubits (has to be same length as self._num_qubits)
            q_ancillas (list): list of ancilla qubits (or None if none needed)
            params (list): parameters for circuit
        """
        raise NotImplementedError()

    def build_inverse(self, qc, q, q_ancillas=None):
        """ Adds inverse of corresponding sub-circuit to given circuit

        Args:
            qc (QuantumCircuit): quantum circuit
            q (list): list of qubits (has to be same length as self._num_qubits)
            q_ancillas (list): list of ancilla qubits (or None if none needed)
        """
        qc_ = QuantumCircuit(*qc.qregs)

        self.build(qc_, q, q_ancillas)
        qc.extend(qc_.inverse())

    def build_controlled(self, qc, q, q_control, q_ancillas=None, use_basis_gates=True):
        """ Adds corresponding controlled sub-circuit to given circuit

        Args:
            qc (QuantumCircuit): quantum circuit
            q (list): list of qubits (has to be same length as self._num_qubits)
            q_control (Qubit): control qubit
            q_ancillas (list): list of ancilla qubits (or None if none needed)
            use_basis_gates (bool): use basis gates for expansion of controlled circuit
        """
        uncontrolled_circuit = QuantumCircuit(*qc.qregs)
        self.build(uncontrolled_circuit, q, q_ancillas)

        controlled_circuit = get_controlled_circuit(uncontrolled_circuit,
                                                    q_control, use_basis_gates=use_basis_gates)
        qc.extend(controlled_circuit)

    def build_controlled_inverse(self, qc, q, q_control, q_ancillas=None, use_basis_gates=True):
        """ Adds controlled inverse of corresponding sub-circuit to given circuit

        Args:
            qc (QuantumCircuit): quantum circuit
            q (list): list of qubits (has to be same length as self._num_qubits)
            q_control (Qubit): control qubit
            q_ancillas (list): list of ancilla qubits (or None if none needed)
            use_basis_gates (bool): use basis gates for expansion of controlled circuit
        """
        qc_ = QuantumCircuit(*qc.qregs)

        self.build_controlled(qc_, q, q_control, q_ancillas, use_basis_gates)
        qc.extend(qc_.inverse())

    def build_power(self, qc, q, power, q_ancillas=None):
        """ Adds power of corresponding circuit.
            May be overridden if a more efficient implementation is possible """
        for _ in range(power):
            self.build(qc, q, q_ancillas)

    def build_inverse_power(self, qc, q, power, q_ancillas=None):
        """ Adds inverse power of corresponding circuit.
            May be overridden if a more efficient implementation is possible """
        for _ in range(power):
            self.build_inverse(qc, q, q_ancillas)

    def build_controlled_power(self, qc, q, q_control, power,
                               q_ancillas=None, use_basis_gates=True):
        """ Adds controlled power of corresponding circuit.
            May be overridden if a more efficient implementation is possible """
        for _ in range(power):
            self.build_controlled(qc, q, q_control, q_ancillas, use_basis_gates)

    def build_controlled_inverse_power(self, qc, q, q_control, power,
                                       q_ancillas=None, use_basis_gates=True):
        """ Adds controlled, inverse, power of corresponding circuit.
            May be overridden if a more efficient implementation is possible """
        for _ in range(power):
            self.build_controlled_inverse(qc, q, q_control, q_ancillas, use_basis_gates)
