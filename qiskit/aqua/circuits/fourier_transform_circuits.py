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
from qiskit.aqua.utils.circuit_utils import is_qubit_list


class FourierTransformCircuits:

    @staticmethod
    def _do_swaps(circuit, qubits):
        num_qubits = len(qubits)
        for i in range(num_qubits // 2):
            circuit.cx(qubits[i], qubits[num_qubits - i - 1])
            circuit.cx(qubits[num_qubits - i - 1], qubits[i])
            circuit.cx(qubits[i], qubits[num_qubits - i - 1])

    @staticmethod
    def construct_circuit(
            circuit=None,
            qubits=None,
            inverse=False,
            approximation_degree=0,
            do_swaps=True
    ):
        """
        Construct the circuit representing the desired state vector.

        Args:
            circuit (QuantumCircuit): The optional circuit to extend from.
            qubits (QuantumRegister | list of qubits): The optional qubits to construct the circuit with.
            approximation_degree (int): degree of approximation for the desired circuit
            inverse (bool): Boolean flag to indicate Inverse Quantum Fourier Transform
            do_swaps (bool): Boolean flag to specify if swaps should be included to align the qubit order of
                input and output. The output qubits would be in reversed order without the swaps.

        Returns:
            QuantumCircuit.
        """

        if circuit is None:
            raise AquaError('Missing input QuantumCircuit.')

        if qubits is None:
            raise AquaError('Missing input qubits.')
        else:
            if isinstance(qubits, QuantumRegister):
                if not circuit.has_register(qubits):
                    circuit.add_register(qubits)
            elif is_qubit_list(qubits):
                for qubit in qubits:
                    if not circuit.has_register(qubit[0]):
                        circuit.add_register(qubit[0])
            else:
                raise AquaError('A QuantumRegister or a list of qubits is expected for the input qubits.')

        if do_swaps and not inverse:
            FourierTransformCircuits._do_swaps(circuit, qubits)

        qubit_range = reversed(range(len(qubits))) if inverse else range(len(qubits))
        for j in qubit_range:
            neighbor_range = range(np.max([0, j - len(qubits) + approximation_degree + 1]), j)
            if inverse:
                neighbor_range = reversed(neighbor_range)
                circuit.u2(0, np.pi, qubits[j])
            for k in neighbor_range:
                lam = 1.0 * np.pi / float(2 ** (j - k))
                if inverse:
                    lam *= -1
                circuit.u1(lam / 2, qubits[j])
                circuit.cx(qubits[j], qubits[k])
                circuit.u1(-lam / 2, qubits[k])
                circuit.cx(qubits[j], qubits[k])
                circuit.u1(lam / 2, qubits[k])
            if not inverse:
                circuit.u2(0, np.pi, qubits[j])

        if do_swaps and inverse:
            FourierTransformCircuits._do_swaps(circuit, qubits)

        return circuit
