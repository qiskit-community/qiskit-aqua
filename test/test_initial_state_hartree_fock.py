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

from test.common import QiskitAquaChemistryTestCase
from qiskit_aqua_chemistry.aqua_extensions.components.initial_states import HartreeFock


class TestInitialStateHartreeFock(QiskitAquaChemistryTestCase):

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


if __name__ == '__main__':
    unittest.main()
