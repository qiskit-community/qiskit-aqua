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

from qiskit.circuit import QuantumRegister, QuantumCircuit, Qubit

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

    def construct_circuit(self, circuit=None, qubits=None):
        """
        Construct the circuit representing the desired state vector.

        Args:
            circuit (QuantumCircuit): The optional circuit to extend from.
            qubits (QuantumRegister | list of Qubit): The optional qubits to construct the circuit with.

        Returns:
            QuantumCircuit.
        """

        if qubits is None:
            qubits = QuantumRegister(self._num_qubits, name='q')

        # in case `qubits` is a list of Qubits
        if isinstance(qubits, list):
            # create empty circuit if necessary
            if circuit is None:
                circuit = QuantumCircuit()
            # loop over all qubits and add the required registers
            for q in qubits:
                if not isinstance(q, Qubit):
                    raise AquaError('Unexpected element type {} in qubit list.'.format(type(q)))
                if not circuit.has_register(q.register):
                    circuit.add_register(q.register)
            # construct state initialization circuit
            temp = QuantumCircuit(*circuit.qregs)

        # otherwise, if it is a QuantumRegister
        elif isinstance(qubits, QuantumRegister):
            if circuit is None:
                circuit = QuantumCircuit(qubits)
            else:
                if not circuit.has_register(qubits):
                    circuit.add_register(qubits)
            temp = QuantumCircuit(qubits)

        else:
            raise AquaError('Unexpected qubits type {}.'.format(type(qubits)))

        if len(qubits) < self._num_qubits:
            raise AquaError('Insufficient qubits are provided for the intended state-vector.')

        temp.initialize(self._state_vector, [qubits[i] for i in range(self._num_qubits)])
        temp = convert_to_basis_gates(temp)
        # remove the reset gates terra's unroller added
        temp.data = [g for g in temp.data if not g[0].name == 'reset']
        circuit += temp
        return circuit
