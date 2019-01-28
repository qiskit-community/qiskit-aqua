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
from qiskit.quantum_info import Pauli
from qiskit.aqua import Operator

from test.common import QiskitAquaChemistryTestCase
from qiskit.chemistry.bksf import edge_operator_aij, edge_operator_bi


class TestBKSFMapping(QiskitAquaChemistryTestCase):

    def test_bksf_edge_op_bi(self):
        """Test bksf mapping, edge operator bi"""
        edge_matrix = np.triu(np.ones((4, 4)))
        edge_list = np.array(np.nonzero(np.triu(edge_matrix) - np.diag(np.diag(edge_matrix))))
        qterm_b0 = edge_operator_bi(edge_list, 0)
        qterm_b1 = edge_operator_bi(edge_list, 1)
        qterm_b2 = edge_operator_bi(edge_list, 2)
        qterm_b3 = edge_operator_bi(edge_list, 3)

        ref_qterm_b0 = Operator(paulis=[[1.0, Pauli.from_label('IIIZZZ')]])
        ref_qterm_b1 = Operator(paulis=[[1.0, Pauli.from_label('IZZIIZ')]])
        ref_qterm_b2 = Operator(paulis=[[1.0, Pauli.from_label('ZIZIZI')]])
        ref_qterm_b3 = Operator(paulis=[[1.0, Pauli.from_label('ZZIZII')]])

        self.assertEqual(qterm_b0, ref_qterm_b0, "\n{} vs \n{}".format(
            qterm_b0.print_operators(), ref_qterm_b0.print_operators()))
        self.assertEqual(qterm_b1, ref_qterm_b1, "\n{} vs \n{}".format(
            qterm_b1.print_operators(), ref_qterm_b1.print_operators()))
        self.assertEqual(qterm_b2, ref_qterm_b2, "\n{} vs \n{}".format(
            qterm_b2.print_operators(), ref_qterm_b2.print_operators()))
        self.assertEqual(qterm_b3, ref_qterm_b3, "\n{} vs \n{}".format(
            qterm_b3.print_operators(), ref_qterm_b3.print_operators()))

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

        ref_qterm_a01 = Operator(paulis=[[1.0, Pauli.from_label('IIIIIX')]])
        ref_qterm_a02 = Operator(paulis=[[1.0, Pauli.from_label('IIIIXZ')]])
        ref_qterm_a03 = Operator(paulis=[[1.0, Pauli.from_label('IIIXZZ')]])
        ref_qterm_a12 = Operator(paulis=[[1.0, Pauli.from_label('IIXIZZ')]])
        ref_qterm_a13 = Operator(paulis=[[1.0, Pauli.from_label('IXZZIZ')]])
        ref_qterm_a23 = Operator(paulis=[[1.0, Pauli.from_label('XZZZZI')]])

        self.assertEqual(qterm_a01, ref_qterm_a01, "\n{} vs \n{}".format(
            qterm_a01.print_operators(), ref_qterm_a01.print_operators()))
        self.assertEqual(qterm_a02, ref_qterm_a02, "\n{} vs \n{}".format(
            qterm_a02.print_operators(), ref_qterm_a02.print_operators()))
        self.assertEqual(qterm_a03, ref_qterm_a03, "\n{} vs \n{}".format(
            qterm_a03.print_operators(), ref_qterm_a03.print_operators()))
        self.assertEqual(qterm_a12, ref_qterm_a12, "\n{} vs \n{}".format(
            qterm_a12.print_operators(), ref_qterm_a12.print_operators()))
        self.assertEqual(qterm_a13, ref_qterm_a13, "\n{} vs \n{}".format(
            qterm_a13.print_operators(), ref_qterm_a13.print_operators()))
        self.assertEqual(qterm_a23, ref_qterm_a23, "\n{} vs \n{}".format(
            qterm_a23.print_operators(), ref_qterm_a23.print_operators()))


if __name__ == '__main__':
    unittest.main()
