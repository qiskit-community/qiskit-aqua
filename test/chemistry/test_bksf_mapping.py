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

""" Test BKSF Mapping """

import unittest
from test.chemistry import QiskitChemistryTestCase
import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliSumOp
from qiskit.chemistry.bksf import edge_operator_aij, edge_operator_bi


class TestBKSFMapping(QiskitChemistryTestCase):
    """ BKSF Mapping tests """

    def test_bksf_edge_op_bi(self):
        """Test bksf mapping, edge operator bi"""
        edge_matrix = np.triu(np.ones((4, 4)))
        edge_list = np.array(np.nonzero(np.triu(edge_matrix) - np.diag(np.diag(edge_matrix))))
        qterm_b0 = edge_operator_bi(edge_list, 0)
        qterm_b1 = edge_operator_bi(edge_list, 1)
        qterm_b2 = edge_operator_bi(edge_list, 2)
        qterm_b3 = edge_operator_bi(edge_list, 3)

        ref_qterm_b0 = PauliSumOp.from_list([('IIIZZZ', 1)])
        ref_qterm_b1 = PauliSumOp.from_list([('IZZIIZ', 1)])
        ref_qterm_b2 = PauliSumOp.from_list([('ZIZIZI', 1)])
        ref_qterm_b3 = PauliSumOp.from_list([('ZZIZII', 1)])

        self.assertEqual(qterm_b0, ref_qterm_b0)
        self.assertEqual(qterm_b1, ref_qterm_b1)
        self.assertEqual(qterm_b2, ref_qterm_b2)
        self.assertEqual(qterm_b3, ref_qterm_b3)

    def test_bksf_edge_op_aij(self):
        """Test bksf mapping, edge operator aij"""
        edge_matrix = np.triu(np.ones((4, 4)))
        edge_list = np.array(np.nonzero(np.triu(edge_matrix) - np.diag(np.diag(edge_matrix))))
        qterm_a01 = edge_operator_aij(edge_list, 0, 1)
        qterm_a02 = edge_operator_aij(edge_list, 0, 2)
        qterm_a03 = edge_operator_aij(edge_list, 0, 3)
        qterm_a12 = edge_operator_aij(edge_list, 1, 2)
        qterm_a13 = edge_operator_aij(edge_list, 1, 3)
        qterm_a23 = edge_operator_aij(edge_list, 2, 3)

        ref_qterm_a01 = PauliSumOp.from_list([('IIIIIX', 1)])
        ref_qterm_a02 = PauliSumOp.from_list([('IIIIXZ', 1)])
        ref_qterm_a03 = PauliSumOp.from_list([('IIIXZZ', 1)])
        ref_qterm_a12 = PauliSumOp.from_list([('IIXIZZ', 1)])
        ref_qterm_a13 = PauliSumOp.from_list([('IXZZIZ', 1)])
        ref_qterm_a23 = PauliSumOp.from_list([('XZZZZI', 1)])

        self.assertEqual(qterm_a01, ref_qterm_a01)
        self.assertEqual(qterm_a02, ref_qterm_a02)
        self.assertEqual(qterm_a03, ref_qterm_a03)
        self.assertEqual(qterm_a12, ref_qterm_a12)
        self.assertEqual(qterm_a13, ref_qterm_a13)
        self.assertEqual(qterm_a23, ref_qterm_a23)


if __name__ == '__main__':
    unittest.main()
