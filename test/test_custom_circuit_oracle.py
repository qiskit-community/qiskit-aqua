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

import unittest
from qiskit import BasicAer, QuantumCircuit, QuantumRegister
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.oracles import CustomCircuitOracle
from qiskit.aqua.algorithms import DeutschJozsa
from test.common import QiskitAquaTestCase


class TestCustomCircuitOracle(QiskitAquaTestCase):

    def test_using_dj_with_constant_func(self):
        qv = QuantumRegister(2, name='v')
        qo = QuantumRegister(1, name='o')
        circuit = QuantumCircuit(qv, qo)
        circuit.x(qo[0])

        oracle = CustomCircuitOracle(variable_register=qv, output_register=qo, circuit=circuit)
        algorithm = DeutschJozsa(oracle)
        result = algorithm.run(quantum_instance=QuantumInstance(BasicAer.get_backend('qasm_simulator')))
        self.assertTrue(result['result'] == 'constant')

    def test_using_dj_with_balanced_func(self):
        qv = QuantumRegister(2, name='v')
        qo = QuantumRegister(1, name='o')
        circuit = QuantumCircuit(qv, qo)
        circuit.cx(qv[0], qo[0])

        oracle = CustomCircuitOracle(variable_register=qv, output_register=qo, circuit=circuit)
        algorithm = DeutschJozsa(oracle)
        result = algorithm.run(quantum_instance=QuantumInstance(BasicAer.get_backend('qasm_simulator')))
        self.assertTrue(result['result'] == 'balanced')


if __name__ == '__main__':
    unittest.main()
