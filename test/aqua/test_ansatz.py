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

"""Tests for Aqua's Ansatz object."""

from ddt import ddt, data, unpack

from qiskit import QuantumCircuit
from qiskit.circuit.random.utils import random_circuit
from qiskit.extensions.standard import XGate, RXGate, CrxGate

from qiskit.aqua.components.ansatz import Ansatz

from test.aqua import QiskitAquaTestCase


@ddt
class TestAnsatz(QiskitAquaTestCase):
    def setUp(self):
        super().setUp()

    def test_empty_ansatz(self):
        ansatz = Ansatz()
        print(ansatz)
        self.assertEqual(ansatz.num_qubits, 0)
        self.assertEqual(ansatz.num_parameters, 0)

        self.assertEqual(ansatz.to_circuit(), QuantumCircuit())

        for attribute in [ansatz._gates, ansatz._qargs, ansatz._reps]:
            self.assertEqual(len(attribute), 0)

    @data(
        [(XGate(), [0])],
        [(XGate(), [0]), (XGate(), [2])],
        [(RXGate(0.2), [2]), (CrxGate(-0.2), [1, 3])],
    )
    def test_append_gates_to_empty_ansatz(self, gate_data):
        ansatz = Ansatz()

        max_num_qubits = 0
        for (_, indices) in gate_data:
            max_num_qubits = max(max_num_qubits, max(indices))

        reference = QuantumCircuit(max_num_qubits + 1)
        for (gate, indices) in gate_data:
            # ansatz.append(gate, indices)
            reference.append(gate, indices)

        self.assertEqual(ansatz.to_circuit(), reference)

    @data(
        [5, 3], [1, 5]
    )
    def test_append_circuit(self, num_qubits):
        # fixed depth of 3 gates per circuit
        depth = 3

        # keep track of a reference circuit
        reference = QuantumCircuit(max(num_qubits) + 1)

        # construct the Ansatz from the first circuit
        first_circuit = random_circuit(num_qubits[0], depth)
        ansatz = Ansatz(first_circuit.to_gate())
        reference.append(first_circuit, list(range(num_qubits[0])))

        # append the rest
        for num in num_qubits[1:]:
            circuit = random_circuit(num, depth)
            ansatz.append(circuit)
            reference.append(circuit, list(range(num)))

        self.assertEqual(ansatz.to_circuit(), reference)
