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

""" Test Custom Circuit Oracle """

import unittest
from test.aqua import QiskitAquaTestCase
from qiskit import BasicAer, QuantumCircuit, QuantumRegister
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.components.oracles import CustomCircuitOracle
from qiskit.aqua.algorithms import DeutschJozsa
from qiskit.aqua.algorithms import Grover


class TestCustomCircuitOracle(QiskitAquaTestCase):
    """ Test Custom Circuit Oracle """
    def test_using_dj_with_constant_func(self):
        """ using dj with constant func test """
        q_v = QuantumRegister(2, name='v')
        q_o = QuantumRegister(1, name='o')
        circuit = QuantumCircuit(q_v, q_o)
        circuit.x(q_o[0])

        oracle = CustomCircuitOracle(variable_register=q_v, output_register=q_o, circuit=circuit)
        algorithm = DeutschJozsa(oracle)
        result = algorithm.run(
            quantum_instance=QuantumInstance(BasicAer.get_backend('qasm_simulator')))
        self.assertEqual(result['result'], 'constant')

    def test_using_dj_with_balanced_func(self):
        """ using dj with balanced func test """
        q_v = QuantumRegister(2, name='v')
        q_o = QuantumRegister(1, name='o')
        circuit = QuantumCircuit(q_v, q_o)
        circuit.cx(q_v[0], q_o[0])

        oracle = CustomCircuitOracle(variable_register=q_v, output_register=q_o, circuit=circuit)
        algorithm = DeutschJozsa(oracle)
        result = algorithm.run(
            quantum_instance=QuantumInstance(BasicAer.get_backend('qasm_simulator')))
        self.assertEqual(result['result'], 'balanced')

    def test_using_grover_for_error(self):
        """ using grover without providing evaluate_classically callback """
        q_v = QuantumRegister(2, name='v')
        q_o = QuantumRegister(1, name='o')
        circuit = QuantumCircuit(q_v, q_o)
        oracle = CustomCircuitOracle(variable_register=q_v, output_register=q_o, circuit=circuit)
        with self.assertRaises(AquaError):
            _ = Grover(oracle)

    def test_using_grover_for_ccx(self):
        """ using grover correctly (with the evaluate_classically callback provided) """
        q_v = QuantumRegister(2, name='v')
        q_o = QuantumRegister(1, name='o')
        circuit = QuantumCircuit(q_v, q_o)
        circuit.ccx(q_v[0], q_v[1], q_o[0])
        oracle = CustomCircuitOracle(variable_register=q_v, output_register=q_o, circuit=circuit,
                                     evaluate_classically_callback=lambda m: (m == '11', [1, 2]))
        algorithm = Grover(oracle)
        result = algorithm.run(
            quantum_instance=QuantumInstance(BasicAer.get_backend('qasm_simulator')))
        self.assertEqual(result.assignment, [1, 2])


if __name__ == '__main__':
    unittest.main()
