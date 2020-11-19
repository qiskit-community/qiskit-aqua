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

""" Test Initial State Zero """

import unittest
import warnings
from test.aqua import QiskitAquaTestCase
import numpy as np
from qiskit.aqua.components.initial_states import Zero


class TestInitialStateZero(QiskitAquaTestCase):
    """ Test Initial State Zero """

    def setUp(self):
        super().setUp()
        warnings.filterwarnings('ignore', category=DeprecationWarning)

    def tearDown(self):
        super().tearDown()
        warnings.filterwarnings('always', category=DeprecationWarning)

    def test_qubits_2_vector(self):
        """ Qubits 2 vector test """
        zero = Zero(2)
        cct = zero.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [1.0, 0.0, 0.0, 0.0])

    def test_qubits_5_vector(self):
        """ Qubits 5 vector test """
        zero = Zero(5)
        cct = zero.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_qubits_2_circuit(self):
        """ Qubits 2 Circuit test """
        zero = Zero(2)
        cct = zero.construct_circuit('circuit')
        # pylint: disable=no-member
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n')

    def test_qubits_5_circuit(self):
        """ Qubits 5 circuit test """
        zero = Zero(5)
        cct = zero.construct_circuit('circuit')
        # pylint: disable=no-member
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\n')


if __name__ == '__main__':
    unittest.main()
