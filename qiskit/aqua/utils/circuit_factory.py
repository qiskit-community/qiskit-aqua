# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Abstract CircuitFactory to build a circuit, along with inverse, controlled
and power combinations of the circuit.
"""

from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.aqua import AquaError
from qiskit.aqua.utils.controlledcircuit import get_controlled_circuit


class CircuitFactory(ABC):

    """ Base class for CircuitFactories """

    def __init__(self, num_target_qubits):
        self._num_target_qubits = num_target_qubits
        pass

    @property
    def num_target_qubits(self):
        """ Returns the number of target qubits """
        return self._num_target_qubits

    def required_ancillas(self):
        return 0

    def required_ancillas_controlled(self):
        return 0

    def get_num_qubits(self):
        return self._num_target_qubits + self.required_ancillas()

    def get_num_qubits_controlled(self):
        return self._num_target_qubits + self.required_ancillas_controlled()

    @abstractmethod
    def build(self, qc, q, q_ancillas=None, params=None):
        """ Adds corresponding sub-circuit to given circuit

        Args:
            qc : quantum circuit
            q : list of qubits (has to be same length as self._num_qubits)
            q_ancillas : list of ancilla qubits (or None if none needed)
            params : parameters for circuit
        """
        raise NotImplementedError()

    def build_inverse(self, qc, q, q_ancillas=None, params=None):
        """ Adds inverse of corresponding sub-circuit to given circuit

        Args:
            qc : quantum circuit
            q : list of qubits (has to be same length as self._num_qubits)
            q_ancillas : list of ancilla qubits (or None if none needed)
            params : parameters for circuit
        """
        qc_ = QuantumCircuit(*qc.qregs)

        self.build(qc_, q, q_ancillas, params)
        qc.extend(qc_.inverse())

    def build_controlled(self, qc, q, q_control, q_ancillas=None, params=None):
        """ Adds corresponding controlled sub-circuit to given circuit

        Args:
            qc : quantum circuit
            q : list of qubits (has to be same length as self._num_qubits)
            q_control : control qubit
            q_ancillas : list of ancilla qubits (or None if none needed)
            params : parameters for circuit
        """
        uncontrolled_circuit = QuantumCircuit(*qc.qregs)

        self.build(uncontrolled_circuit, q, q_ancillas, params)
        controlled_circuit = get_controlled_circuit(uncontrolled_circuit, q_control)
        qc.extend(controlled_circuit)

    def build_controlled_inverse(self, qc, q, q_control, q_ancillas=None, params=None):
        """ Adds controlled inverse of corresponding sub-circuit to given circuit

        Args:
            qc : quantum circuit
            q : list of qubits (has to be same length as self._num_qubits)
            q_control : control qubit
            q_ancillas : list of ancilla qubits (or None if none needed)
            params : parameters for circuit
        """
        qc_ = QuantumCircuit(*qc.qregs)

        self.build_controlled(qc_, q, q_control, q_ancillas, params)
        qc.extend(qc_.inverse())

    def build_power(self, qc, q, power, q_ancillas=None, params=None):
        """ Adds power of corresponding circuit.
            May be overridden if a more efficient implementation is possible """
        for _ in range(power):
            self.build(qc, q, q_ancillas, params)

    def build_inverse_power(self, qc, q, power, q_ancillas=None, params=None):
        """ Adds inverse power of corresponding circuit.
            May be overridden if a more efficient implementation is possible """
        for _ in range(power):
            self.build_inverse(qc, q, q_ancillas, params)

    def build_controlled_power(self, qc, q, q_control, power, q_ancillas=None, params=None):
        """ Adds controlled power of corresponding circuit.
            May be overridden if a more efficient implementation is possible """
        for _ in range(power):
            self.build_controlled(qc, q, q_control, q_ancillas, params)

    def build_controlled_inverse_power(self, qc, q, q_control, power, q_ancillas=None, params=None):
        """ Adds controlled, inverse, power of corresponding circuit.
            May be overridden if a more efficient implementation is possible """
        for _ in range(power):
            self.build_controlled_inverse(qc, q, q_control, q_ancillas, params)
