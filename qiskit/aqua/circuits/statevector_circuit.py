# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
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

        # in case the register is a list of qubits
        if type(register) is list:

            # create empty circuit if necessary
            if circuit is None:
                circuit = QuantumCircuit()

            # loop over all qubits and add the required registers
            for q in register:
                if not circuit.has_register(q[0]):
                    circuit.add_register(q[0])

            # construct state initialization circuit
            temp = QuantumCircuit(*circuit.qregs)

        # otherwise, if it is a real register
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
        # remove the reset gates terra's unroller added
        temp.data = [g for g in temp.data if not g[0].name == 'reset']
        circuit += temp
        return circuit
