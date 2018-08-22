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

import copy
import unittest
from collections import OrderedDict

import numpy as np
from qiskit.tools.qi.pauli import label_to_pauli
from qiskit_aqua import Operator
from qiskit_aqua.utils import random_unitary

from test.common import QiskitAquaChemistryTestCase
from qiskit_aqua_chemistry import FermionicOperator
from qiskit_aqua_chemistry.bksf import edge_operator_aij, edge_operator_bi
from qiskit_aqua_chemistry.drivers import ConfigurationManager


def h2_transform_slow(h2, unitary_matrix):
    """
    Transform h2 based on unitry matrix, and overwrite original property.
    #MARK: A naive implementation based on MATLAB implementation.
    Args:
        unitary_matrix (numpy 2-D array, np.float or np.complex):
                    Unitary matrix for h2 transformation.
    """
    num_modes = unitary_matrix.shape[0]
    temp1 = np.zeros((num_modes, num_modes, num_modes, num_modes), dtype=unitary_matrix.dtype)
    temp2 = np.zeros((num_modes, num_modes, num_modes, num_modes), dtype=unitary_matrix.dtype)
    temp3 = np.zeros((num_modes, num_modes, num_modes, num_modes), dtype=unitary_matrix.dtype)
    temp_ret = np.zeros((num_modes, num_modes, num_modes, num_modes), dtype=unitary_matrix.dtype)
    unitary_matrix_dagger = np.conjugate(unitary_matrix)
    for a in range(num_modes):
        for i in range(num_modes):
            temp1[a, :, :, :] += unitary_matrix_dagger[i, a] * h2[i, :, :, :]
        for b in range(num_modes):
            for j in range(num_modes):
                temp2[a, b, :, :] += unitary_matrix[j, b] * temp1[a, j, :, :]
            for c in range(num_modes):
                for k in range(num_modes):
                    temp3[a, b, c, :] += unitary_matrix_dagger[k, c] * temp2[a, b, k, :]
                for d in range(num_modes):
                    for l in range(num_modes):
                        temp_ret[a, b, c, d] += unitary_matrix[l, d] * temp3[a, b, c, l]
    return temp_ret


class TestFermionicOperator(QiskitAquaChemistryTestCase):
    """Fermionic Operator tests."""

    def setUp(self):
        cfg_mgr = ConfigurationManager()
        pyscf_cfg = OrderedDict([('atom', 'Li .0 .0 .0; H .0 .0 1.595'), ('unit', 'Angstrom'),
                                 ('charge', 0), ('spin', 0), ('basis', 'sto3g')])
        section = {}
        section['properties'] = pyscf_cfg
        driver = cfg_mgr.get_driver_instance('PYSCF')
        molecule = driver.run(section)
        self.fer_op = FermionicOperator(h1=molecule._one_body_integrals,
                                        h2=molecule._two_body_integrals)

    def test_transform(self):
        unitary_matrix = random_unitary(self.fer_op.h1.shape[0])

        reference_fer_op = copy.deepcopy(self.fer_op)
        target_fer_op = copy.deepcopy(self.fer_op)

        reference_fer_op._h1_transform(unitary_matrix)
        reference_fer_op.h2 = h2_transform_slow(reference_fer_op.h2, unitary_matrix)

        target_fer_op._h1_transform(unitary_matrix)
        target_fer_op._h2_transform(unitary_matrix)

        h1_nonzeros = np.count_nonzero(reference_fer_op.h1 - target_fer_op.h1)
        self.assertEqual(h1_nonzeros, 0, "there are differences between h1 transformation")

        h2_nonzeros = np.count_nonzero(reference_fer_op.h2 - target_fer_op.h2)
        self.assertEqual(h2_nonzeros, 0, "there are differences between h2 transformation")

    def test_freezing_core(self):
        cfg_mgr = ConfigurationManager()
        pyscf_cfg = OrderedDict([('atom', 'H .0 .0 -1.160518; Li .0 .0 0.386839'),
                                 ('unit', 'Angstrom'), ('charge', 0),
                                 ('spin', 0), ('basis', 'sto3g')])
        section = {}
        section['properties'] = pyscf_cfg
        driver = cfg_mgr.get_driver_instance('PYSCF')
        molecule = driver.run(section)
        fer_op = FermionicOperator(h1=molecule._one_body_integrals,
                                   h2=molecule._two_body_integrals)
        fer_op, energy_shift = fer_op.fermion_mode_freezing([0, 6])
        gt = -7.8187092970493755
        diff = abs(energy_shift - gt)
        self.assertLess(diff, 1e-6)

        cfg_mgr = ConfigurationManager()
        pyscf_cfg = OrderedDict([('atom', 'H .0 .0 .0; Na .0 .0 1.888'), ('unit', 'Angstrom'),
                                 ('charge', 0), ('spin', 0), ('basis', 'sto3g')])
        section = {}
        section['properties'] = pyscf_cfg
        driver = cfg_mgr.get_driver_instance('PYSCF')
        molecule = driver.run(section)
        fer_op = FermionicOperator(h1=molecule._one_body_integrals,
                                   h2=molecule._two_body_integrals)
        fer_op, energy_shift = fer_op.fermion_mode_freezing([0, 1, 2, 3, 4, 10, 11, 12, 13, 14])
        gt = -162.58414559586748
        diff = abs(energy_shift - gt)
        self.assertLess(diff, 1e-6)

    def test_bksf_mapping(self):
        """Test bksf mapping

        The spectrum of bksf mapping should be half of jordan wigner mapping.
        """
        cfg_mgr = ConfigurationManager()
        pyscf_cfg = OrderedDict([('atom', 'H .0 .0 0.7414; H .0 .0 .0'), ('unit', 'Angstrom'),
                                 ('charge', 0), ('spin', 0), ('basis', 'sto3g')])
        section = {}
        section['properties'] = pyscf_cfg
        driver = cfg_mgr.get_driver_instance('PYSCF')
        molecule = driver.run(section)
        fer_op = FermionicOperator(h1=molecule._one_body_integrals,
                                   h2=molecule._two_body_integrals)
        jw_op = fer_op.mapping('jordan_wigner')
        bksf_op = fer_op.mapping('bravyi_kitaev_sf')
        jw_op.to_matrix()
        bksf_op.to_matrix()
        jw_eigs = np.linalg.eigvals(jw_op.matrix.toarray())
        bksf_eigs = np.linalg.eigvals(bksf_op.matrix.toarray())

        jw_eigs = np.sort(np.around(jw_eigs.real, 6))
        bksf_eigs = np.sort(np.around(bksf_eigs.real, 6))
        overlapped_spectrum = np.sum(np.isin(jw_eigs, bksf_eigs))

        self.assertEqual(overlapped_spectrum, jw_eigs.size // 2)


class TestBKSFMapping(QiskitAquaChemistryTestCase):

    def test_bksf_edge_op_bi(self):
        """Test bksf mapping, edge operator bi"""
        edge_matrix = np.triu(np.ones((4, 4)))
        edge_list = np.array(np.nonzero(np.triu(edge_matrix) - np.diag(np.diag(edge_matrix))))
        qterm_b0 = edge_operator_bi(edge_list, 0)
        qterm_b1 = edge_operator_bi(edge_list, 1)
        qterm_b2 = edge_operator_bi(edge_list, 2)
        qterm_b3 = edge_operator_bi(edge_list, 3)

        ref_qterm_b0 = Operator(paulis=[[1.0, label_to_pauli('ZZZIII')]])
        ref_qterm_b1 = Operator(paulis=[[1.0, label_to_pauli('ZIIZZI')]])
        ref_qterm_b2 = Operator(paulis=[[1.0, label_to_pauli('IZIZIZ')]])
        ref_qterm_b3 = Operator(paulis=[[1.0, label_to_pauli('IIZIZZ')]])

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

        ref_qterm_a01 = Operator(paulis=[[1.0, label_to_pauli('XIIIII')]])
        ref_qterm_a02 = Operator(paulis=[[1.0, label_to_pauli('ZXIIII')]])
        ref_qterm_a03 = Operator(paulis=[[1.0, label_to_pauli('ZZXIII')]])
        ref_qterm_a12 = Operator(paulis=[[1.0, label_to_pauli('ZZIXII')]])
        ref_qterm_a13 = Operator(paulis=[[1.0, label_to_pauli('ZIZZXI')]])
        ref_qterm_a23 = Operator(paulis=[[1.0, label_to_pauli('IZZZZX')]])

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
