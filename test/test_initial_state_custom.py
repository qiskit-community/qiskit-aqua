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

import unittest

import numpy as np

from qiskit.aqua import AquaError
from qiskit.aqua.components.initial_states import Custom
from test.common import QiskitAquaTestCase


class TestInitialStateCustom(QiskitAquaTestCase):

    def test_qubits_2_zero_vector(self):
        self.custom = Custom(2, state='zero')
        cct = self.custom.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [1.0, 0.0, 0.0, 0.0])

    def test_qubits_5_zero_vector(self):
        self.custom = Custom(5, state='zero')
        cct = self.custom.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_qubits_2_zero_circuit(self):
        self.custom = Custom(2, state='zero')
        cct = self.custom.construct_circuit('circuit')
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n')

    def test_qubits_5_zero_circuit(self):
        self.custom = Custom(5, state='zero')
        cct = self.custom.construct_circuit('circuit')
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\n')

    def test_qubits_2_uniform_vector(self):
        self.custom = Custom(2, state='uniform')
        cct = self.custom.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.5]*4)

    def test_qubits_5_uniform_vector(self):
        self.custom = Custom(5, state='uniform')
        cct = self.custom.construct_circuit('vector')
        np.testing.assert_array_almost_equal(cct, [0.1767767]*32)

    def test_qubits_2_uniform_circuit(self):
        self.custom = Custom(2, state='uniform')
        cct = self.custom.construct_circuit('circuit')
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
                                     'u2(0.0,3.14159265358979) q[0];\nu2(0.0,3.14159265358979) q[1];\n')

    def test_qubits_2_random_vector(self):
        self.custom = Custom(2, state='random')
        cct = self.custom.construct_circuit('vector')
        prob = np.sqrt(np.sum([x**2 for x in cct]))
        self.assertAlmostEqual(prob, 1.0)

    def test_qubits_5_random_vector(self):
        self.custom = Custom(5, state='random')
        cct = self.custom.construct_circuit('vector')
        prob = np.sqrt(np.sum([x**2 for x in cct]))
        self.assertAlmostEqual(prob, 1.0)

    def test_qubits_2_given_vector(self):
        self.custom = Custom(2, state_vector=[0.5]*4)
        cct = self.custom.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.5]*4)

    def test_qubits_5_given_vector(self):
        self.custom = Custom(5, state_vector=[1.0]*32)
        cct = self.custom.construct_circuit('vector')
        np.testing.assert_array_almost_equal(cct, [0.1767767]*32)

    def test_qubits_5_randgiven_vector(self):
        self.custom = Custom(5, state_vector=np.random.rand(32))
        cct = self.custom.construct_circuit('vector')
        prob = np.sqrt(np.sum([x**2 for x in cct]))
        self.assertAlmostEqual(prob, 1.0)

    def test_qubits_qubits_given_mistmatch(self):
        with self.assertRaises(AquaError):
            self.custom = Custom(5, state_vector=[1.0]*23)

    def test_qubits_2_zero_vector_wrong_cct_mode(self):
        self.custom = Custom(5, state='zero')
        with self.assertRaises(AquaError):
            cct = self.custom.construct_circuit('matrix')


if __name__ == '__main__':
    unittest.main()
