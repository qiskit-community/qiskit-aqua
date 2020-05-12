# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Shor """

import unittest
import math
import operator
from test.aqua import QiskitAquaTestCase
from ddt import ddt, data, idata, unpack
from qiskit import BasicAer, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Shor


@ddt
class TestShor(QiskitAquaTestCase):
    """test Shor's algorithm"""

    @idata([
        [15, 'qasm_simulator', [3, 5]],
    ])
    @unpack
    def test_shor_factoring(self, n_v, backend, factors):
        """ shor factoring test """
        shor = Shor(n_v)
        result_dict = shor.run(QuantumInstance(BasicAer.get_backend(backend), shots=1000))
        self.assertListEqual(result_dict['factors'][0], factors)

    @data(5, 7)
    def test_shor_no_factors(self, n_v):
        """ shor no factors test """
        shor = Shor(n_v)
        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1000)
        ret = shor.run(quantum_instance)
        self.assertTrue(ret['factors'] == [])

    @idata([
        [3, 5],
        [5, 3],
    ])
    @unpack
    def test_shor_power(self, base, power):
        """ shor power test """
        n_v = int(math.pow(base, power))
        shor = Shor(n_v)
        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1000)
        ret = shor.run(quantum_instance)
        self.assertTrue(ret['factors'] == [base])

    @data(-1, 0, 1, 2, 4, 16)
    def test_shor_bad_input(self, n_v):
        """ shor bad input test """
        with self.assertRaises(ValueError):
            Shor(n_v)

    @idata([[2, 15, 8]])
    @unpack
    def test_shor_modinv(self, a_v, m_v, expected):
        """ shor modular inverse test """
        modinv = Shor.modinv(a_v, m_v)
        self.assertTrue(modinv == expected)

    @idata([[3, "0011"],
            [5, "0101"]])
    @unpack
    def test_phi_add_gate(self, addition_magnitude, expected_state):
        """ shor phi add gate test """
        shor = Shor(3)
        shor._n = 2
        shor._qft.num_qubits = 3
        shor._iqft.num_qubits = 3
        q = QuantumRegister(4)
        c = ClassicalRegister(4, name='measurement')
        circuit = QuantumCircuit(q, c)

        gate = shor._phi_add_gate(3, addition_magnitude)
        qubits = [q[i] for i in reversed(range(len(q) - 1))]

        circuit.compose(shor._qft, qubits, inplace=True)
        circuit.compose(gate, qubits, inplace=True)
        circuit.compose(shor._iqft, qubits, inplace=True)
        circuit.measure(q, c)

        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1000)

        result = quantum_instance.execute(circuit)

        result_data = result.get_counts().items()
        most_likely_state = max(result_data, key=operator.itemgetter(1))[0]
        self.assertTrue(most_likely_state, expected_state)


if __name__ == '__main__':
    unittest.main()
