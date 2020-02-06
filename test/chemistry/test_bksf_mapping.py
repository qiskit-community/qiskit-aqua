# -*- coding: utf-8 -*-

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
from qiskit.aqua.operators import WeightedPauliOperator
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

        ref_qterm_b0 = WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('IIIZZZ')]])
        ref_qterm_b1 = WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('IZZIIZ')]])
        ref_qterm_b2 = WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('ZIZIZI')]])
        ref_qterm_b3 = WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('ZZIZII')]])

        self.assertEqual(qterm_b0, ref_qterm_b0, "\n{} vs \n{}".format(
            qterm_b0.print_details(), ref_qterm_b0.print_details()))
        self.assertEqual(qterm_b1, ref_qterm_b1, "\n{} vs \n{}".format(
            qterm_b1.print_details(), ref_qterm_b1.print_details()))
        self.assertEqual(qterm_b2, ref_qterm_b2, "\n{} vs \n{}".format(
            qterm_b2.print_details(), ref_qterm_b2.print_details()))
        self.assertEqual(qterm_b3, ref_qterm_b3, "\n{} vs \n{}".format(
            qterm_b3.print_details(), ref_qterm_b3.print_details()))

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

        ref_qterm_a01 = WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('IIIIIX')]])
        ref_qterm_a02 = WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('IIIIXZ')]])
        ref_qterm_a03 = WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('IIIXZZ')]])
        ref_qterm_a12 = WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('IIXIZZ')]])
        ref_qterm_a13 = WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('IXZZIZ')]])
        ref_qterm_a23 = WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('XZZZZI')]])

        self.assertEqual(qterm_a01, ref_qterm_a01, "\n{} vs \n{}".format(
            qterm_a01.print_details(), ref_qterm_a01.print_details()))
        self.assertEqual(qterm_a02, ref_qterm_a02, "\n{} vs \n{}".format(
            qterm_a02.print_details(), ref_qterm_a02.print_details()))
        self.assertEqual(qterm_a03, ref_qterm_a03, "\n{} vs \n{}".format(
            qterm_a03.print_details(), ref_qterm_a03.print_details()))
        self.assertEqual(qterm_a12, ref_qterm_a12, "\n{} vs \n{}".format(
            qterm_a12.print_details(), ref_qterm_a12.print_details()))
        self.assertEqual(qterm_a13, ref_qterm_a13, "\n{} vs \n{}".format(
            qterm_a13.print_details(), ref_qterm_a13.print_details()))
        self.assertEqual(qterm_a23, ref_qterm_a23, "\n{} vs \n{}".format(
            qterm_a23.print_details(), ref_qterm_a23.print_details()))


if __name__ == '__main__':
    unittest.main()
