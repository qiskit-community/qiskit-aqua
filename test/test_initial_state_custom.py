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

import unittest
import numpy as np

from test.common import QiskitAquaTestCase
from qiskit_aqua import get_initial_state_instance


class TestInitialStateCustom(QiskitAquaTestCase):

    def setUp(self):
        self.custom = get_initial_state_instance('CUSTOM')

    def test_qubits_2_zero_vector(self):
        self.custom.init_args(2, state='zero')
        cct = self.custom.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [1.0, 0.0, 0.0, 0.0])

    def test_qubits_5_zero_vector(self):
        self.custom.init_args(5, state='zero')
        cct = self.custom.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_qubits_2_zero_circuit(self):
        self.custom.init_args(2, state='zero')
        cct = self.custom.construct_circuit('circuit')
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n')

    def test_qubits_5_zero_circuit(self):
        self.custom.init_args(5, state='zero')
        cct = self.custom.construct_circuit('circuit')
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\n')

    def test_qubits_2_uniform_vector(self):
        self.custom.init_args(2, state='uniform')
        cct = self.custom.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.5]*4)

    def test_qubits_5_uniform_vector(self):
        self.custom.init_args(5, state='uniform')
        cct = self.custom.construct_circuit('vector')
        np.testing.assert_array_almost_equal(cct, [0.1767767]*32)

    def test_qubits_2_uniform_circuit(self):
        self.custom.init_args(2, state='uniform')
        cct = self.custom.construct_circuit('circuit')
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
                                     'u2(0.0,3.14159265358979) q[0];\nu2(0.0,3.14159265358979) q[1];\n')

    def test_qubits_2_random_vector(self):
        self.custom.init_args(2, state='random')
        cct = self.custom.construct_circuit('vector')
        prob = np.sqrt(np.sum([x**2 for x in cct]))
        self.assertAlmostEqual(prob, 1.0)

    def test_qubits_5_random_vector(self):
        self.custom.init_args(5, state='random')
        cct = self.custom.construct_circuit('vector')
        prob = np.sqrt(np.sum([x**2 for x in cct]))
        self.assertAlmostEqual(prob, 1.0)

    def test_qubits_2_given_vector(self):
        self.custom.init_args(2, state_vector=[0.5]*4)
        cct = self.custom.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.5]*4)

    def test_qubits_5_given_vector(self):
        self.custom.init_args(5, state_vector=[1.0]*32)
        cct = self.custom.construct_circuit('vector')
        np.testing.assert_array_almost_equal(cct, [0.1767767]*32)

    def test_qubits_5_randgiven_vector(self):
        self.custom.init_args(5, state_vector=np.random.rand(32))
        cct = self.custom.construct_circuit('vector')
        prob = np.sqrt(np.sum([x**2 for x in cct]))
        self.assertAlmostEqual(prob, 1.0)

    def test_qubits_qubits_given_mistmatch(self):
        with self.assertRaises(ValueError):
            self.custom.init_args(5, state_vector=[1.0]*23)

    def test_qubits_2_zero_vector_wrong_cct_mode(self):
        self.custom.init_args(2, state='zero')
        with self.assertRaises(ValueError):
            cct = self.custom.construct_circuit('matrix')


if __name__ == '__main__':
    unittest.main()
