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
import copy
from collections import OrderedDict

import numpy as np

from qiskit_aqua_chemistry import FermionicOperator
from qiskit_aqua.utils import random_unitary
from test.common import QiskitAquaChemistryTestCase
from qiskit_aqua_chemistry.drivers import ConfigurationManager

# def mapping_slow(self, map_type, threshold=0.00000001):
#     """
#     Args:
#         map_type (str): case-insensitive mapping type. "jordan_wigner", "parity", "bravyi_kitaev"
#         threshold (float): threshold for Pauli simplification
#     Returns:
#         Operator Class: create an Operator object in Paulis form.
#     """

#     """
#     ####################################################################
#     ############   DEFINING MAPPED FERMIONIC OPERATORS    ##############
#     ####################################################################
#     """
#     n = len(self._h1)  # number of fermionic modes / qubits
#     map_type = map_type.lower()
#     if map_type == 'jordan_wigner':
#         a = self._jordan_wigner_mode(n)
#     elif map_type == 'parity':
#         a = self._parity_mode(n)
#     elif map_type == 'bravyi_kitaev':
#         a = self._bravyi_kitaev_mode(n)
#     else:
#         raise AquaChemistryError('Please specify the supported modes: jordan_wigner, parity, bravyi_kitaev')
#     """
#     ####################################################################
#     ############    BUILDING THE MAPPED HAMILTONIAN     ################
#     ####################################################################
#     """
#     pauli_list = Operator(paulis=[])
#     """
#     #######################    One-body    #############################
#     """
#     for i in range(n):
#         for j in range(n):
#             if self._h1[i, j] != 0:
#                 for alpha in range(2):
#                     for beta in range(2):
#                         pauli_prod = sgn_prod(a[i][alpha], a[j][beta])
#                         pauli_term = [self._h1[i, j] * 1 / 4 * pauli_prod[1] *
#                                       np.power(-1j, alpha) *
#                                       np.power(1j, beta),
#                                       pauli_prod[0]]
#                         if np.absolute(pauli_term[0]) > threshold:
#                             pauli_list += Operator(paulis=[pauli_term])
#     pauli_list.chop(threshold=threshold)
#     """
#     #######################    Two-body    #############################
#     """
#     for i in range(n):
#         for j in range(n):
#             for k in range(n):
#                 for m in range(n):
#                     if self._h2[i, j, k, m] != 0:
#                         for alpha in range(2):
#                             for beta in range(2):
#                                 for gamma in range(2):
#                                     for delta in range(2):
#                                         """
#                                         # Note: chemists' notation for the
#                                         # labeling,
#                                         # h2(i,j,k,m) adag_i adag_k a_m a_j
#                                         """
#                                         pauli_prod_1 = sgn_prod(
#                                             a[i][alpha], a[k][beta])
#                                         pauli_prod_2 = sgn_prod(
#                                             pauli_prod_1[0], a[m][gamma])
#                                         pauli_prod_3 = sgn_prod(
#                                             pauli_prod_2[0], a[j][delta])

#                                         phase1 = pauli_prod_1[1] * \
#                                             pauli_prod_2[1] * pauli_prod_3[1]
#                                         phase2 = np.power(-1j, alpha + beta) * \
#                                             np.power(1j, gamma + delta)

#                                         pauli_term = [
#                                             self._h2[i, j, k, m] / 16 * phase1 *
#                                             phase2, pauli_prod_3[0]]
#                                         if np.absolute(pauli_term[0]) > threshold:
#                                             pauli_list += Operator(paulis=[pauli_term])
#     pauli_list.chop(threshold=threshold)

#     if self._ph_trans_shift is not None:
#         pauli_list += Operator(paulis=[[self._ph_trans_shift, label_to_pauli('I' * self._h1.shape[0])]])

#     return pauli_list

def h2_transform_slow(h2, unitary_matrix):
    """
    Transform h2 based on unitry matrix, and overwrite original property.
    #MARK: A naive implementation based on MATLAB implementation.
    Args:
        unitary_matrix (numpy 2-D array, np.float or np.complex): Unitary matrix for h2 transformation.
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
        pyscf_cfg = OrderedDict([('atom', 'Li .0 .0 .0; H .0 .0 1.595'), ('unit', 'Angstrom'), ('charge', 0), ('spin', 0), ('basis', 'sto3g')])
        section = {}
        section['properties'] = pyscf_cfg
        driver = cfg_mgr.get_driver_instance('PYSCF')
        molecule = driver.run(section)
        self.ferOp = FermionicOperator(h1=molecule._one_body_integrals, h2=molecule._two_body_integrals)

    def test_transform(self):
        unitary_matrix = random_unitary(self.ferOp.h1.shape[0])

        reference_ferOp = copy.deepcopy(self.ferOp)
        target_ferOp = copy.deepcopy(self.ferOp)

        reference_ferOp._h1_transform(unitary_matrix)
        reference_ferOp.h2 = h2_transform_slow(reference_ferOp.h2, unitary_matrix)

        target_ferOp._h1_transform(unitary_matrix)
        target_ferOp._h2_transform(unitary_matrix)

        h1_nonzeros = np.count_nonzero(reference_ferOp.h1 - target_ferOp.h1)
        self.assertEqual(h1_nonzeros, 0, "there are differences between h1 transformation")

        h2_nonzeros = np.count_nonzero(reference_ferOp.h2 - target_ferOp.h2)
        self.assertEqual(h2_nonzeros, 0, "there are differences between h2 transformation")


    # @parameterized.expand([
    #     ['jordan_wigner'],
    #     ['parity'],
    #     ['bravyi_kitaev']
    # ])
    # def test_mapping(self, map_type):
    #     ref_jwQubitOp = self.ferOp.mapping_slow(map_type=map_type, threshold=1e-10)
    #     tar_jwQubitOp = self.ferOp.mapping(map_type=map_type, threshold=1e-10)

    #     ref_jwQubitOp.convert("paulis", "matrix")
    #     tar_jwQubitOp.convert("paulis", "matrix")

    #     eqaulity = ref_jwQubitOp.matrix - tar_jwQubitOp.matrix
    #     self.assertLess(abs(eqaulity).mean(), 1e-10, "there are differences between mapped qubit operator")

    def test_freezing_core(self):
        cfg_mgr = ConfigurationManager()
        pyscf_cfg = OrderedDict([('atom', 'H .0 .0 -1.160518; Li .0 .0 0.386839'), ('unit', 'Angstrom'), ('charge', 0), ('spin', 0), ('basis', 'sto3g')])
        section = {}
        section['properties'] = pyscf_cfg
        driver = cfg_mgr.get_driver_instance('PYSCF')
        molecule = driver.run(section)
        ferOp = FermionicOperator(h1=molecule._one_body_integrals, h2=molecule._two_body_integrals)
        ferOp, energy_shift = ferOp.fermion_mode_freezing([0, 6])
        gt = -7.8187092970493755
        diff = abs(energy_shift - gt)
        self.assertLess(diff, 1e-6)

        cfg_mgr = ConfigurationManager()
        pyscf_cfg = OrderedDict([('atom', 'H .0 .0 .0; Na .0 .0 1.888'), ('unit', 'Angstrom'), ('charge', 0), ('spin', 0), ('basis', 'sto3g')])
        section = {}
        section['properties'] = pyscf_cfg
        driver = cfg_mgr.get_driver_instance('PYSCF')
        molecule = driver.run(section)
        ferOp = FermionicOperator(h1=molecule._one_body_integrals, h2=molecule._two_body_integrals)
        ferOp, energy_shift = ferOp.fermion_mode_freezing([0, 1, 2, 3, 4, 10, 11, 12, 13, 14])
        gt = -162.58414559586748
        diff = abs(energy_shift - gt)
        self.assertLess(diff, 1e-6)

if __name__ == '__main__':
    unittest.main()
