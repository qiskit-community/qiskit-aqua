# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Operator ExpectationValue """

import unittest
import itertools
import os
from test.aqua.common import QiskitAquaTestCase
import numpy as np
from parameterized import parameterized
from qiskit import BasicAer, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli, state_fidelity
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.operators import WeightedPauliOperator, op_converter
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.initial_states import Custom


class TestExpectationValue(QiskitAquaTestCase):
    """Operator ExpectationValue tests."""

    def setUp(self):
        super().setUp()
        seed = 0
        aqua_globals.random_seed = seed

        self.num_qubits = 3
        paulis = [Pauli.from_label(pauli_label)
                  for pauli_label in itertools.product('IXYZ', repeat=self.num_qubits)]
        weights = aqua_globals.random.random_sample(len(paulis))
        self.qubit_op = WeightedPauliOperator.from_list(paulis, weights)
        self.var_form = RYRZ(self.qubit_op.num_qubits, 1)

        qasm_simulator = BasicAer.get_backend('qasm_simulator')
        self.quantum_instance_qasm = QuantumInstance(qasm_simulator, shots=65536,
                                                     seed_simulator=seed, seed_transpiler=seed)
        statevector_simulator = BasicAer.get_backend('statevector_simulator')
        self.quantum_instance_statevector = \
            QuantumInstance(statevector_simulator, shots=1,
                            seed_simulator=seed, seed_transpiler=seed)

    def test_from_to_file(self):
        """ from to file test """
        paulis = [Pauli.from_label(x) for x in ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']]
        weights = [0.2 + -1j * 0.8, 0.6 + -1j * 0.6, 0.8 + -1j * 0.2,
                   -0.2 + -1j * 0.8, -0.6 - -1j * 0.6, -0.8 - -1j * 0.2]
        op = WeightedPauliOperator.from_list(paulis, weights)
        file_path = self._get_resource_path('temp_op.json')
        op.to_file(file_path)
        self.assertTrue(os.path.exists(file_path))

        load_op = WeightedPauliOperator.from_file(file_path)
        self.assertEqual(op, load_op)
        os.remove(file_path)
