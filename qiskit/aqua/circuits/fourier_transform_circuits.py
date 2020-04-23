# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""DEPRECATED. Quantum Fourier Transform Circuit."""

import warnings

from qiskit.circuit.library import QFT
from qiskit.aqua import AquaError


class FourierTransformCircuits:
    """DEPRECATED. Quantum Fourier Transform Circuit."""

    @staticmethod
    def construct_circuit(
            circuit=None,
            qubits=None,
            inverse=False,
            approximation_degree=0,
            do_swaps=True
    ):
        """Construct the circuit representing the desired state vector.

        Args:
            circuit (QuantumCircuit): The optional circuit to extend from.
            qubits (Union(QuantumRegister, list[Qubit])): The optional qubits to construct
                the circuit with.
            approximation_degree (int): degree of approximation for the desired circuit
            inverse (bool): Boolean flag to indicate Inverse Quantum Fourier Transform
            do_swaps (bool): Boolean flag to specify if swaps should be included to align
                the qubit order of
                input and output. The output qubits would be in reversed order without the swaps.

        Returns:
            QuantumCircuit: quantum circuit
        Raises:
            AquaError: invalid input
        """
        warnings.warn('The class FourierTransformCircuits is deprecated and will be removed '
                      'no earlier than 3 months after the release 0.7.0. You should use the '
                      'qiskit.circuit.library.QFT class instead.',
                      DeprecationWarning, stacklevel=2)

        if circuit is None:
            raise AquaError('Missing input QuantumCircuit.')

        if qubits is None:
            raise AquaError('Missing input qubits.')

        qft = QFT(len(qubits), approximation_degree=approximation_degree, do_swaps=do_swaps)
        if inverse:
            qft = qft.inverse()

        circuit.append(qft.to_instruction(), qubits)

        return circuit
