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

from test.common import QiskitAquaTestCase
from qiskit.aqua.components.initial_states import Zero


class TestInitialStateZero(QiskitAquaTestCase):

    def test_qubits_2_vector(self):
        self.zero = Zero(2)
        cct = self.zero.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [1.0, 0.0, 0.0, 0.0])

    def test_qubits_5_vector(self):
        self.zero = Zero(5)
        cct = self.zero.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_qubits_2_circuit(self):
        self.zero = Zero(2)
        cct = self.zero.construct_circuit('circuit')
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n')

    def test_qubits_5_circuit(self):
        self.zero = Zero(5)
        cct = self.zero.construct_circuit('circuit')
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\n')


if __name__ == '__main__':
    unittest.main()
