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

""" Test Initial State Custom """

import warnings
import unittest
from test.aqua import QiskitAquaTestCase

import numpy as np

from qiskit.aqua import AquaError, aqua_globals
from qiskit.aqua.components.initial_states import Custom


class TestInitialStateCustom(QiskitAquaTestCase):
    """ Test Initial State Custom """

    def setUp(self):
        super().setUp()
        warnings.filterwarnings('ignore', category=DeprecationWarning)

    def tearDown(self):
        super().tearDown()
        warnings.filterwarnings('always', category=DeprecationWarning)

    def test_qubits_2_zero_vector(self):
        """ qubits 2 zero vector test """
        custom = Custom(2, state='zero')
        cct = custom.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [1.0, 0.0, 0.0, 0.0])

    def test_qubits_5_zero_vector(self):
        """ qubits 5 zero vector test """
        custom = Custom(5, state='zero')
        cct = custom.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_qubits_2_zero_circuit(self):
        """ qubits 2 zero circuit test """
        custom = Custom(2, state='zero')
        cct = custom.construct_circuit('circuit')
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n')

    def test_qubits_5_zero_circuit(self):
        """ qubits 5 zero circuit test """
        custom = Custom(5, state='zero')
        cct = custom.construct_circuit('circuit')
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\n')

    def test_qubits_2_uniform_vector(self):
        """ qubits 2 uniform vector test """
        custom = Custom(2, state='uniform')
        cct = custom.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.5] * 4)

    def test_qubits_5_uniform_vector(self):
        """ qubits 5 uniform vector test """
        custom = Custom(5, state='uniform')
        cct = custom.construct_circuit('vector')
        np.testing.assert_array_almost_equal(cct, [0.1767767] * 32)

    def test_qubits_2_uniform_circuit(self):
        """ qubits 2 uniform circuit test """
        custom = Custom(2, state='uniform')
        cct = custom.construct_circuit('circuit')
        self.assertEqual(cct.qasm(),
                         'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
                         'h q[0];\nh q[1];\n')

    def test_qubits_2_random_vector(self):
        """ qubits 2 random vector test """
        custom = Custom(2, state='random')
        cct = custom.construct_circuit('vector')
        prob = np.sqrt(np.sum([x**2 for x in cct]))
        self.assertAlmostEqual(prob, 1.0)

    def test_qubits_5_random_vector(self):
        """ qubits 5 random vector test """
        custom = Custom(5, state='random')
        cct = custom.construct_circuit('vector')
        prob = np.sqrt(np.sum([x**2 for x in cct]))
        self.assertAlmostEqual(prob, 1.0)

    def test_qubits_2_given_vector(self):
        """ qubits 2 given vector test """
        custom = Custom(2, state_vector=[0.5] * 4)
        cct = custom.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.5] * 4)

    def test_qubits_5_given_vector(self):
        """ qubits 5 given vector test """
        custom = Custom(5, state_vector=[1.0] * 32)
        cct = custom.construct_circuit('vector')
        np.testing.assert_array_almost_equal(cct, [0.1767767] * 32)

    def test_qubits_5_randgiven_vector(self):
        """ qubits 5 randgiven vector test """
        aqua_globals.random_seed = 32
        custom = Custom(5, state_vector=aqua_globals.random.random(32))
        cct = custom.construct_circuit('vector')
        prob = np.sqrt(np.sum([x**2 for x in cct]))
        self.assertAlmostEqual(prob, 1.0)

    def test_qubits_qubits_given_mismatch(self):
        """ qubits 5 given mismatch test """
        with self.assertRaises(AquaError):
            _ = Custom(5, state_vector=[1.0] * 23)

    def test_qubits_2_zero_vector_wrong_cct_mode(self):
        """ qubits 2 zero vector wrong cct mode test """
        custom = Custom(5, state='zero')
        with self.assertRaises(AquaError):
            _ = custom.construct_circuit('matrix')


if __name__ == '__main__':
    unittest.main()
