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

from test.common import QiskitChemistryTestCase
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock


class TestInitialStateHartreeFock(QiskitChemistryTestCase):

    def test_qubits_4_jw_h2(self):
        self.hf = HartreeFock(4, 4, 2, 'jordan_wigner', False)
        cct = self.hf.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_qubits_4_py_h2(self):
        self.hf = HartreeFock(4, 4, 2, 'parity', False)
        cct = self.hf.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_qubits_4_bk_h2(self):
        self.hf = HartreeFock(4, 4, 2, 'bravyi_kitaev', False)
        cct = self.hf.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_qubits_2_py_h2(self):
        self.hf = HartreeFock(2, 4, 2, 'parity', True)
        cct = self.hf.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.0, 1.0, 0.0, 0.0])

    def test_qubits_2_py_h2_cct(self):
        self.hf = HartreeFock(2, 4, 2, 'parity', True)
        cct = self.hf.construct_circuit('circuit')
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
                                     'u3(3.14159265358979,0.0,3.14159265358979) q[0];\n')

    def test_qubits_6_py_lih_cct(self):
        self.hf = HartreeFock(6, 10, 2, 'parity', True, [1, 2])
        cct = self.hf.construct_circuit('circuit')
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[6];\n'
                                     'u3(3.14159265358979,0.0,3.14159265358979) q[0];\n'
                                     'u3(3.14159265358979,0.0,3.14159265358979) q[1];\n')


if __name__ == '__main__':
    unittest.main()
