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
Quantum Fourier Transform Circuit.
"""

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua import AquaError


class FourierTransformCircuits:

    def __init__(self, num_qubits, approximation_degree=0, inverse=False):
        """Constructor.

        Args:
            num_qubits (int): number of qubits for the fourier transform circuit
            approximation_degree (int): degree of approximation for the desired circuit
            inverse (bool): Boolean flag to indicate Inverse Quantum Fourier Transform
        """
        self._num_qubits = num_qubits
        self._approximation_degree = approximation_degree
        self._inverse = inverse

    @staticmethod
    def _set_up(circ, qubits, num_qubits):
        if circ:
            if not qubits:
                raise AquaError(
                    'A QuantumRegister or a list of qubits need to be specified with the input QuantumCircuit.'
                )
        else:
            circ = QuantumCircuit()
            if not qubits:
                qubits = QuantumRegister(num_qubits, name='q')

        if len(qubits) < num_qubits:
            raise AquaError('Insufficient input qubits: {} provided but {} needed.'.format(
                len(qubits), num_qubits
            ))

        if isinstance(qubits, QuantumRegister):
            _ = qubits
        elif isinstance(qubits, list) and isinstance(qubits[0], tuple) and isinstance(qubits[0][0], QuantumRegister):
            _ = qubits[0][0]
        else:
            raise AquaError('Unrecognized input. Register or qubits expected.')
        if not circ.has_register(_):
            circ.add_register(_)
        return circ, qubits

    def construct_circuit(self, qubits=None, circuit=None):
        """
        Construct the circuit representing the desired state vector.

        Args:
            qubits (QuantumRegister | list of qubits): The optional qubits to construct the circuit with.
            circuit (QuantumCircuit): The optional circuit to extend from.

        Returns:
            QuantumCircuit.
        """
        circuit, qubits = FourierTransformCircuits._set_up(circuit, qubits, self._num_qubits)

        qubit_range = range(self._num_qubits)
        if self._inverse:
            qubit_range = reversed(qubit_range)
        for j in qubit_range:
            neighbor_range = range(np.max([0, j - self._num_qubits + self._approximation_degree + 1]), j)
            if self._inverse:
                neighbor_range = reversed(neighbor_range)
                circuit.u2(0, np.pi, qubits[j])
            for k in neighbor_range:
                lam = 1.0 * np.pi / float(2 ** (j - k))
                if self._inverse:
                    lam *= -1
                circuit.u1(lam / 2, qubits[j])
                circuit.cx(qubits[j], qubits[k])
                circuit.u1(-lam / 2, qubits[k])
                circuit.cx(qubits[j], qubits[k])
                circuit.u1(lam / 2, qubits[k])
            if not self._inverse:
                circuit.u2(0, np.pi, qubits[j])
        return circuit
