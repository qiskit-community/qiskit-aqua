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
    """
    Arbitrary State-Vector Circuit.
    """
    def __init__(self, state_vector):
        """Constructor.

        Args:
            state_vector (numpy.ndarray): vector representation of the desired quantum state
        Raises:
            AquaError: invalid input
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
            register (Union(QuantumRegister , list[Qubit])): The optional qubits to construct
                            the circuit with.

        Returns:
            QuantumCircuit: quantum circuit
        Raises:
            AquaError: invalid input
        """

        if register is None:
            register = QuantumRegister(self._num_qubits, name='q')

        # in case `register` is a list of Qubits
        if isinstance(register, list):
            # create empty circuit if necessary
            if circuit is None:
                circuit = QuantumCircuit()
            # loop over all register and add the required registers
            for q in register:
                if not isinstance(q, Qubit):
                    raise AquaError('Unexpected element type {} in qubit list.'.format(type(q)))
                if not circuit.has_register(q.register):
                    circuit.add_register(q.register)
            # construct state initialization circuit
            temp = QuantumCircuit(*circuit.qregs)

        # otherwise, if it is a QuantumRegister
        elif isinstance(register, QuantumRegister):
            if circuit is None:
                circuit = QuantumCircuit(register)
            else:
                if not circuit.has_register(register):
                    circuit.add_register(register)
            temp = QuantumCircuit(register)

        else:
            raise AquaError('Unexpected register type {}.'.format(type(register)))

        if len(register) < self._num_qubits:
            raise AquaError('Insufficient register are provided for the intended state-vector.')

        temp.initialize(self._state_vector, [register[i] for i in range(self._num_qubits)])
        temp = convert_to_basis_gates(temp)
        # remove the reset gates terra's unroller added
        temp.data = [g for g in temp.data if not g[0].name == 'reset']
        circuit += temp
        return circuit
