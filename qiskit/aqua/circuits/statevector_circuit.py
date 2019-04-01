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
Arbitrary State-Vector Circuit.
"""

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua import AquaError
from qiskit.aqua.utils.arithmetic import normalize_vector, is_power_of_2, log2
from qiskit.aqua.utils.circuit_utils import convert_to_basis_gates


class StateVectorCircuit:

    def __init__(self, state_vector):
        """Constructor.

        Args:
            state_vector: vector representation of the desired quantum state
        """
        if not is_power_of_2(len(state_vector)):
            raise AquaError('The length of the input state vector needs to be a power of 2.')
        self._num_qubits = log2(len(state_vector))
        self._state_vector = normalize_vector(state_vector)

    def construct_circuit(self, circuit=None, register=None):
        """
        Construct the circuit representing the desired state vector.

        Args:
            circuit (QuantumCircuit): The optional circuit to extend from.
            register (QuantumRegister): The optional register to construct the circuit with.

        Returns:
            QuantumCircuit.
        """
        if register is None:
            register = QuantumRegister(self._num_qubits, name='q')
        else:
            if len(register) < self._num_qubits:
                raise AquaError('The provided register does not have enough qubits.')

        if circuit is None:
            circuit = QuantumCircuit(register)
        else:
            if not circuit.has_register(register):
                circuit.add_register(register)

        # TODO: add capability to start in the middle of the register
        temp = QuantumCircuit(register)
        temp.initialize(self._state_vector, [register[i] for i in range(self._num_qubits)])
        temp = convert_to_basis_gates(temp)
        circuit += temp
        return circuit
