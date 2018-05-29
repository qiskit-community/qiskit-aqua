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

import concurrent.futures
import multiprocessing
import itertools
import numpy as np
import logging
from qiskit.tools.qi.pauli import Pauli, sgn_prod, label_to_pauli

from qiskit_acqua import Operator
from qiskit_acqua_chemistry import ACQUAChemistryError
from .particle_hole import particle_hole_transformation

logger = logging.getLogger(__name__)


class FermionicOperator(object):
    """
    A set of functions to map fermionic Hamiltonians to qubit Hamiltonians.

    References:
    - E. Wigner and P. Jordan., Über das Paulische Äguivalenzverbot,
        Z. Phys., 47:631 (1928).
    - S. Bravyi and A. Kitaev. Fermionic quantum computation,
        Ann. of Phys., 298(1):210–226 (2002).
    - A. Tranter, S. Sofia, J. Seeley, M. Kaicher, J. McClean, R. Babbush,
        P. Coveney, F. Mintert, F. Wilhelm, and P. Love. The Bravyi–Kitaev
        transformation: Properties and applications. Int. Journal of Quantum
        Chemistry, 115(19):1431–1441 (2015).
    - S. Bravyi, J. M. Gambetta, A. Mezzacapo, and K. Temme,
        arXiv e-print arXiv:1701.08213 (2017).

    """
    def __init__(self, h1, h2=None, ph_trans_shift=None):
        """
        Args:
            h1 (numpy.ndarray): second-quantized fermionic one-body operator, a 2-D (NxN) tensor
            h2 (numpy.ndarray): second-quantized fermionic two-body operator, a 4-D (NxNxNxN) tensor
            ph_trans_shift (float): energy shift caused by particle hole transformation
        """
        self._h1 = h1
        if h2 is None:
            h2 = np.zeros((h1.shape[0], h1.shape[0], h1.shape[0], h1.shape[0]), dtype=h1.dtype)
        self._h2 = h2
        # self._h1 = COO.from_numpy(h1) if isinstance(h1, numpy.ndarray) else h1
        # if h2 is None:
        #     h2 = np.zeros((h1.shape[0], h1.shape[0], h1.shape[0], h1.shape[0]), dtype=h1.dtype)
        # self._h2 = COO.from_numpy(h2) if isinstance(h2, numpy.ndarray) else h2
        self._ph_trans_shift = ph_trans_shift

    @property
    def h1(self):
        """Getter of one body integral tensor"""
        return self._h1

    @h1.setter
    def h1(self, new_h1):
        """Setter of one body integral tensor"""
        self._h1 = new_h1

    @property
    def h2(self):
        """Getter of two body integral tensor"""
        return self._h2

    @h2.setter
    def h2(self, new_h2):
        """Setter of two body integral tensor"""
        self._h2 = new_h2

    def transform(self, unitary_matrix):
        self._h1_transform(unitary_matrix)
        self._h2_transform(unitary_matrix)

    def _h1_transform(self, unitary_matrix):
        """
        Transform h1 based on unitry matrix, and overwrite original property.
        Args:
            unitary_matrix (numpy.ndarray): A 2-D unitary matrix for h1 transformation.
        """
        self._h1 = unitary_matrix.T.conj().dot(self._h1.dot(unitary_matrix))

    def _h2_transform(self, unitary_matrix):
        """
        Transform h2 to get fermionic hamiltonian, and overwrite original property.

        Args:
            unitary_matrix (numpy.ndarray): A 2-D unitary matrix for h1 transformation.
        """
        num_modes = unitary_matrix.shape[0]
        temp_ret = np.zeros((num_modes, num_modes, num_modes, num_modes), dtype=unitary_matrix.dtype)
        unitary_matrix_dagger = np.conjugate(unitary_matrix)

        # option 1: all temp1, temp2 and temp3 are 4-D tensors.
        # temp1 = np.einsum('ia,i...->...a', unitary_matrix_dagger, h2)
        # temp2 = np.einsum('jb,j...a->...ab', unitary_matrix, temp1)
        # temp3 = np.einsum('kc,k...ab->...abc', unitary_matrix_dagger, temp2)
        # temp_ret = np.einsum('ld,l...abc->...abcd', unitary_matrix, temp3)

        # option 2: temp1 and temp2 are 3-D tensors, temp3 is a 2-D tensor
        # for a in range(num_modes):
        #     temp1 = np.einsum('i,i...->...', unitary_matrix_dagger[:,a], h2)
        #     temp2 = np.einsum('jb,j...->...b', unitary_matrix, temp1)
        #     temp3 = np.einsum('kc,k...b->...bc', unitary_matrix_dagger, temp2)
        #     temp_ret[a,:,:,:] = np.einsum('ld,l...bc->...bcd', unitary_matrix, temp3)

        # option 3: temp1 is a 3-D tensor, temp2 and temp3 are 2-D tensors
        # and this is the fastest option on MacBook 2016.
        for a in range(num_modes):
            temp1 = np.einsum('i,i...->...', unitary_matrix_dagger[:, a], self._h2)
            for b in range(num_modes):
                temp2 = np.einsum('j,j...->...', unitary_matrix[:, b], temp1)
                temp3 = np.einsum('kc,k...->...c', unitary_matrix_dagger, temp2)
                temp_ret[a, b, :, :] = np.einsum('ld,l...c->...cd', unitary_matrix, temp3)

        # option 4: temp1 is 3-D tensor, temp2 and temp3 are 2-D tensor, costs less memory
        # and it is faster than option 1 on MacBook
        # for a in range(num_modes):
        #     temp1 = np.einsum('i,i...->...', unitary_matrix_dagger[:,a], h2)
        #     for b in range(num_modes):
        #         temp2 = np.einsum('j,j...->...', unitary_matrix[:,b], temp1)
        #         for c in range(num_modes):
        #             temp3 = np.einsum('k,k...->...', unitary_matrix_dagger[:,c], temp2)
        #             temp_ret[a,b,c,:] = np.einsum('ld,l...->...d', unitary_matrix, temp3)
        self._h2 = temp_ret

    def _jordan_wigner_mode(self, n):
        """
        Jordan_Wigner mode.

        Args:
            n (int): number of modes
        """
        a = []
        for i in range(n):
            xv = np.asarray([1] * i + [0] + [0] * (n-i-1))
            xw = np.asarray([0] * i + [1] + [0] * (n-i-1))
            yv = np.asarray([1] * i + [1] + [0] * (n-i-1))
            yw = np.asarray([0] * i + [1] + [0] * (n-i-1))

            # xv = np.append(np.append(np.ones(i), 0), np.zeros(n - i - 1))
            # xw = np.append(np.append(np.zeros(i), 1), np.zeros(n - i - 1))
            # yv = np.append(np.append(np.ones(i), 1), np.zeros(n - i - 1))
            # yw = np.append(np.append(np.zeros(i), 1), np.zeros(n - i - 1))
            # defines the two mapped Pauli components of a_i and a_i^\dag,
            # according to a_i -> (a[i][0]+i*a[i][1])/2,
            # a_i^\dag -> (a_[i][0]-i*a[i][1])/2
            a.append((Pauli(xv, xw), Pauli(yv, yw)))

        return a

    def _parity_mode(self, n):
        """
        Parity mode.

        Args:
            n (int): number of modes
        """
        a = []
        for i in range(n):
            Xv = [0] * (i-1) + [1] if i > 0 else []
            Xw = [0] * (i-1) + [0] if i > 0 else []
            Yv = [0] * (i-1) + [0] if i > 0 else []
            Yw = [0] * (i-1) + [0] if i > 0 else []
            Xv = np.asarray(Xv + [0] + [0] * (n-i-1))
            Xw = np.asarray(Xw + [1] + [1] * (n-i-1))
            Yv = np.asarray(Yv + [1] + [0] * (n-i-1))
            Yw = np.asarray(Yw + [1] + [1] * (n-i-1))
            # if i > 1:
            #     Xv = np.append(np.append(np.zeros(i - 1), [1, 0]), np.zeros(n - i - 1))
            #     Xw = np.append(np.append(np.zeros(i - 1), [0, 1]), np.ones(n - i - 1))
            #     Yv = np.append(np.append(np.zeros(i - 1), [0, 1]), np.zeros(n - i - 1))
            #     Yw = np.append(np.append(np.zeros(i - 1), [0, 1]), np.ones(n - i - 1))
            # elif i > 0:
            #     Xv = np.append([1, 0], np.zeros(n - i - 1))
            #     Xw = np.append([0, 1], np.ones(n - i - 1))
            #     Yv = np.append([0, 1], np.zeros(n - i - 1))
            #     Yw = np.append([0, 1], np.ones(n - i - 1))
            # else:
            #     Xv = np.append(0, np.zeros(n - i - 1))
            #     Xw = np.append(1, np.ones(n - i - 1))
            #     Yv = np.append(1, np.zeros(n - i - 1))
            #     Yw = np.append(1, np.ones(n - i - 1))
            # defines the two mapped Pauli components of a_i and a_i^\dag,
            # according to a_i -> (a[i][0]+i*a[i][1])/2,
            # a_i^\dag -> (a_[i][0]-i*a[i][1])/2
            a.append((Pauli(Xv, Xw), Pauli(Yv, Yw)))
        return a

    def _bravyi_kitaev_mode(self, n):
        """
        Bravyi-Kitaev mode

        Args:
            n (int): number of modes
        """
        def parity_set(j, n):
            """Computes the parity set of the j-th orbital in n modes

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indexes

            MARK:
                use `//` to assure the results are integer?
            """
            indexes = np.array([])
            if n % 2 != 0:
                return indexes

            if j < n / 2:
                indexes = np.append(indexes, parity_set(j, n / 2))
            else:
                indexes = np.append(indexes, np.append(
                    parity_set(j - n / 2, n / 2) + n / 2, n / 2 - 1))
            return indexes

        def update_set(j, n):
            """Computes the update set of the j-th orbital in n modes

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indexes

            """
            indexes = np.array([])
            if n % 2 != 0:
                return indexes
            if j < n / 2:
                indexes = np.append(indexes, np.append(
                    n - 1, update_set(j, n / 2)))
            else:
                indexes = np.append(indexes, update_set(j - n / 2, n / 2) + n / 2)
            return indexes

        def flip_set(j, n):
            """Computes the flip set of the j-th orbital in n modes

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indexes

            """
            indexes = np.array([])
            if n % 2 != 0:
                return indexes
            if j < n / 2:
                indexes = np.append(indexes, flip_set(j, n / 2))
            elif j >= n / 2 and j < n - 1:
                indexes = np.append(indexes, flip_set(j - n / 2, n / 2) + n / 2)
            else:
                indexes = np.append(np.append(indexes, flip_set(
                    j - n / 2, n / 2) + n / 2), n / 2 - 1)
            return indexes

        a = []
        # FIND BINARY SUPERSET SIZE
        bin_sup = 1
        while n > np.power(2, bin_sup):
            bin_sup += 1
        # DEFINE INDEX SETS FOR EVERY FERMIONIC MODE
        update_sets = []
        update_pauli = []

        parity_sets = []
        parity_pauli = []

        flip_sets = []

        remainder_sets = []
        remainder_pauli = []
        for j in range(n):

            update_sets.append(update_set(j, np.power(2, bin_sup)))
            update_sets[j] = update_sets[j][update_sets[j] < n]

            parity_sets.append(parity_set(j, np.power(2, bin_sup)))
            parity_sets[j] = parity_sets[j][parity_sets[j] < n]

            flip_sets.append(flip_set(j, np.power(2, bin_sup)))
            flip_sets[j] = flip_sets[j][flip_sets[j] < n]

            remainder_sets.append(np.setdiff1d(parity_sets[j], flip_sets[j]))

            update_pauli.append(Pauli(np.zeros(n), np.zeros(n)))
            parity_pauli.append(Pauli(np.zeros(n), np.zeros(n)))
            remainder_pauli.append(Pauli(np.zeros(n), np.zeros(n)))
            for k in range(n):
                if np.in1d(k, update_sets[j]):
                    update_pauli[j].w[k] = 1
                if np.in1d(k, parity_sets[j]):
                    parity_pauli[j].v[k] = 1
                if np.in1d(k, remainder_sets[j]):
                    remainder_pauli[j].v[k] = 1

            x_j = Pauli(np.zeros(n), np.zeros(n))
            x_j.w[j] = 1
            y_j = Pauli(np.zeros(n), np.zeros(n))
            y_j.v[j] = 1
            y_j.w[j] = 1
            # defines the two mapped Pauli components of a_i and a_i^\dag,
            # according to a_i -> (a[i][0]+i*a[i][1])/2, a_i^\dag ->
            # (a_[i][0]-i*a[i][1])/2
            a.append((update_pauli[j] * x_j * parity_pauli[j],
                      update_pauli[j] * y_j * remainder_pauli[j]))
        return a

    def mapping(self, map_type, threshold=0.00000001, num_workers=4):
        """
        Using multiprocess to speedup the mapping, the improvement can be
        observed when h2 is a non-sparse matrix.

        Args:
            map_type (str): case-insensitive mapping type. "jordan_wigner", "parity", "bravyi_kitaev"
            threshold (float): threshold for Pauli simplification
            num_workers (int): number of processes used to map.

        Returns:
            Operator: create an Operator object in Paulis form.

        Raises:
            ACQUAChemistryError: if the `map_type` can not be recognized.
        """

        """
        ####################################################################
        ############   DEFINING MAPPED FERMIONIC OPERATORS    ##############
        ####################################################################
        """
        n = self._h1.shape[0]  # number of fermionic modes / qubits
        map_type = map_type.lower()
        if map_type == 'jordan_wigner':
            a = self._jordan_wigner_mode(n)
        elif map_type == 'parity':
            a = self._parity_mode(n)
        elif map_type == 'bravyi_kitaev':
            a = self._bravyi_kitaev_mode(n)
        else:
            raise ACQUAChemistryError('Please specify the supported modes: jordan_wigner, parity, bravyi_kitaev')
        """
        ####################################################################
        ############    BUILDING THE MAPPED HAMILTONIAN     ################
        ####################################################################
        """
        max_workers = min(num_workers, multiprocessing.cpu_count())
        pauli_list = Operator(paulis=[])
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            #######################    One-body    #############################
            futures = [executor.submit(FermionicOperator._one_body_mapping, self._h1[i, j], a[i], a[j], threshold)
                       for i, j in itertools.product(range(n), repeat=2) if self._h1[i, j] != 0]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                pauli_list += result
            pauli_list.chop(threshold=threshold)

            #######################    Two-body    #############################
            futures = [executor.submit(FermionicOperator._two_body_mapping,
                                       self._h2[i, j, k, m], a[i], a[j], a[k], a[m], threshold)
                       for i, j, k, m in itertools.product(range(n), repeat=4) if self._h2[i, j, k, m] != 0]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                pauli_list += result
            pauli_list.chop(threshold=threshold)

        if self._ph_trans_shift is not None:
            pauli_list += Operator(paulis=[[self._ph_trans_shift, label_to_pauli('I' * self._h1.shape[0])]])

        return pauli_list

    def mapping_sparse(self, map_type, threshold=0.00000001, num_workers=4):
        """
        Using multiprocess to speedup the mapping, the improvement can be
        observed when h2 is a non-sparse matrix.

        Args:
            map_type (str): case-insensitive mapping type. "jordan_wigner", "parity", "bravyi_kitaev"
            threshold (float): threshold for Pauli simplification
            num_workers (int): number of processes used to map.
        Returns:
            Operator Class: create an Operator object in Paulis form.
        """

        """
        ####################################################################
        ############   DEFINING MAPPED FERMIONIC OPERATORS    ##############
        ####################################################################
        """
        n = self._h1.shape[0]  # number of fermionic modes / qubits
        map_type = map_type.lower()
        if map_type == 'jordan_wigner':
            a = self._jordan_wigner_mode(n)
        elif map_type == 'parity':
            a = self._parity_mode(n)
        elif map_type == 'bravyi_kitaev':
            a = self._bravyi_kitaev_mode(n)
        else:
            raise ACQUAChemistryError('Please specify the supported modes: jordan_wigner, parity, bravyi_kitaev')
        """
        ####################################################################
        ############    BUILDING THE MAPPED HAMILTONIAN     ################
        ####################################################################
        """
        max_workers = min(num_workers, multiprocessing.cpu_count())
        pauli_list = Operator(paulis=[])
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            #######################    One-body    #############################
            futures = [executor.submit(FermionicOperator._one_body_mapping, data, a[i], a[j], threshold)
                       for i, j, data in zip(*self._h1.coords, self._h1.data)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                pauli_list += result
            pauli_list.chop(threshold=threshold)

            #######################    Two-body    #############################
            futures = [executor.submit(FermionicOperator._two_body_mapping, data, a[i], a[j], a[k], a[m], threshold)
                       for i, j, k, m, data in zip(*self._h2.coords, self._h2.data)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                pauli_list += result
            pauli_list.chop(threshold=threshold)

        if self._ph_trans_shift is not None:
            pauli_list += Operator(paulis=[[self._ph_trans_shift, label_to_pauli('I' * self._h1.shape[0])]])

        return pauli_list

    @staticmethod
    def _one_body_mapping(h1_ij, a_i, a_j, threshold):
        """
        Subroutine for one body mapping.

        Args:
            h1_ij (complex): value of h1 at index (i,j)
            a_i (Pauli): pauli at index i
            a_j (Pauli): pauli at index j
            threshold: (float): threshold to remove a pauli

        Returns:
            Operator: Operator for those paulis
        """
        pauli_list = []
        for alpha in range(2):
            for beta in range(2):
                pauli_prod = sgn_prod(a_i[alpha], a_j[beta])
                # pauli_term = [h1_ij / 4 * pauli_prod[1] * \
                #               np.power(-1j, alpha) * \
                #               np.power(1j, beta), \
                #               pauli_prod[0]]
                pauli_term = [h1_ij / 4 * pauli_prod[1] *
                              np.power(1j, 3 * alpha + beta),
                              pauli_prod[0]]
                if np.absolute(pauli_term[0]) > threshold:
                    pauli_list.append(pauli_term)
        return Operator(paulis=pauli_list)

    @staticmethod
    def _two_body_mapping(h2_ijkm, a_i, a_j, a_k, a_m, threshold):
        """
        Subroutine for two body mapping.

        Args:
            h1_ijkm (complex): value of h2 at index (i,j,k,m)
            a_i (Pauli): pauli at index i
            a_j (Pauli): pauli at index j
            a_k (Pauli): pauli at index k
            a_m (Pauli): pauli at index m
            threshold: (float): threshold to remove a pauli

        Returns:
            Operator: Operator for those paulis
        """
        pauli_list = []
        for alpha in range(2):
            for beta in range(2):
                for gamma in range(2):
                    for delta in range(2):
                        pauli_prod_1 = sgn_prod(a_i[alpha], a_k[beta])
                        pauli_prod_2 = sgn_prod(pauli_prod_1[0], a_m[gamma])
                        pauli_prod_3 = sgn_prod(pauli_prod_2[0], a_j[delta])

                        phase1 = pauli_prod_1[1] * pauli_prod_2[1] * pauli_prod_3[1]
                        # phase2 = np.power(-1j, alpha + beta) * np.power(1j, gamma + delta)
                        phase2 = np.power(1j, (3 * (alpha + beta) + gamma + delta) % 4)
                        pauli_term = [h2_ijkm / 16 * phase1 * phase2, pauli_prod_3[0]]
                        if np.absolute(pauli_term[0]) > threshold:
                            pauli_list.append(pauli_term)
        return Operator(paulis=pauli_list)

    def _convert_to_interleaved_spins(self):
        """
        Converting the spin order from up-up-up-up-down-down-down-down
                                    to up-down-up-down-up-down-up-down
        """
        matrix = np.zeros((self._h1.shape), self._h1.dtype)
        N = matrix.shape[0]
        j = np.arange(N//2)
        matrix[j, 2*j] = 1.0
        matrix[j + N // 2, 2*j + 1] = 1.0
        self.transform(matrix)

    def _convert_to_block_spins(self):
        """
        Converting the spin order from up-down-up-down-up-down-up-down
                                    to up-up-up-up-down-down-down-down
        """
        matrix = np.zeros((self._h1.shape), self._h1.dtype)
        N = matrix.shape[0]
        j = np.arange(N//2)
        matrix[2*j, j] = 1.0
        matrix[2*j+1, N//2+j] = 1.0
        self.transform(matrix)

    def particle_hole_transformation(self, num_particles):
        """
        The 'standard' second quantized Hamiltonian can be transformed in the
        particle-hole (p/h) picture, which makes the expansion of the trail wavefunction
        from the HF reference state more natural. In fact, for both trail wavefunctions
        implemented in q-lib ('heuristic' hardware efficient and UCCSD) the p/h Hamiltonian
        improves the speed of convergence of the VQE algorithm for the calculation of
        the electronic ground state properties.
        For more information on the p/h formalism see:
        P. Barkoutsos, arXiv:1805.04340(https://arxiv.org/abs/1805.04340).

        Args:
            num_particles (int): number of particles
        """
        self._convert_to_interleaved_spins()
        h1, h2, energy_shift = particle_hole_transformation(self._h1.shape[0], num_particles, self._h1, self._h2)
        new_ferOp = FermionicOperator(h1=h1, h2=h2, ph_trans_shift=energy_shift)
        new_ferOp._convert_to_block_spins()
        return new_ferOp, energy_shift

    def fermion_mode_elimination(self, fermion_mode_array):
        """
        Generate a new fermionic operator with the modes in fermion_mode_array deleted

        Args:
            fermion_mode_array (list): orbital index for elimination

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        fermion_mode_array = np.sort(fermion_mode_array)
        n_modes_old = self._h1.shape[0]
        n_modes_new = n_modes_old - fermion_mode_array.size
        mode_set_diff = np.setdiff1d(np.arange(n_modes_old), fermion_mode_array)
        h1_id_i, h1_id_j = np.meshgrid(mode_set_diff, mode_set_diff, indexing='ij')
        h1_new = self._h1[h1_id_i, h1_id_j].copy()
        if np.count_nonzero(self._h2) > 0:
            h2_id_i, h2_id_j, h2_id_k, h2_id_l = np.meshgrid(
                mode_set_diff, mode_set_diff, mode_set_diff, mode_set_diff,  indexing='ij')
            h2_new = self._h2[h2_id_i, h2_id_j, h2_id_k, h2_id_l].copy()
        else:
            h2_new = np.zeros((n_modes_new, n_modes_new, n_modes_new, n_modes_new))
        return FermionicOperator(h1_new, h2_new)

    def fermion_mode_freezing(self, fermion_mode_array):
        """
        Generate a fermionic operator with the modes in fermion_mode_array deleted and
        provide the shifted energy after freezing.

        Args:
            fermion_mode_array (list): orbital index for freezing

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        fermion_mode_array = np.sort(fermion_mode_array)
        n_modes_old = self._h1.shape[0]
        n_modes_new = n_modes_old - fermion_mode_array.size
        mode_set_diff = np.setdiff1d(np.arange(n_modes_old), fermion_mode_array)

        h1 = self._h1.copy()
        h2_new = np.zeros((n_modes_new, n_modes_new, n_modes_new, n_modes_new))

        energy_shift = 0.0
        if np.count_nonzero(self._h2) > 0:
            # First simplify h2 and renormalize original h1
            for i, j, l, k in itertools.product(range(n_modes_old), repeat=4):
                # Untouched terms
                h2_ijlk = self._h2[i, j, l, k]
                if h2_ijlk == 0.0:
                        continue
                if (i in mode_set_diff and j in mode_set_diff
                        and l in mode_set_diff and k in mode_set_diff):
                    h2_new[i - np.where(fermion_mode_array < i)[0].size,
                           j - np.where(fermion_mode_array < j)[0].size,
                           l - np.where(fermion_mode_array < l)[0].size,
                           k - np.where(fermion_mode_array < k)[0].size] = h2_ijlk
                else:
                    if i in fermion_mode_array:
                        if l not in fermion_mode_array:
                            if i == k and j not in fermion_mode_array:
                                h1[l, j] -= h2_ijlk
                            elif i == j and k not in fermion_mode_array:
                                h1[l, k] += h2_ijlk
                        elif i != l:
                            if j in fermion_mode_array and i == k and l == j:
                                energy_shift -= h2_ijlk
                            elif l in fermion_mode_array and i == j and l == k:
                                energy_shift += h2_ijlk
                    elif i not in fermion_mode_array and l in fermion_mode_array:
                        if l == k and j not in fermion_mode_array:
                            h1[i, j] += h2_ijlk
                        elif l == j and k not in fermion_mode_array:
                            h1[i, k] -= h2_ijlk

                    # if (i in fermion_mode_array and i == k
                    #         and j not in fermion_mode_array and l not in fermion_mode_array):
                    #     h1[l, j] -= self._h2[i, j, l, k]

                    # elif(i in fermion_mode_array and i == j
                    #      and l not in fermion_mode_array and k not in fermion_mode_array):
                    #     h1[l, k] += self._h2[i, j, l, k]

                    # elif(l in fermion_mode_array and l == k
                    #      and i not in fermion_mode_array and j not in fermion_mode_array):
                    #     h1[i, j] += self._h2[i, j, l, k]

                    # elif(l in fermion_mode_array and l == j
                    #      and i not in fermion_mode_array and k not in fermion_mode_array):
                    #     h1[i, k] -= self._h2[i, j, l, k]

                    # elif(i in fermion_mode_array and j in fermion_mode_array
                    #      and i == k and l == j and i != l):
                    #     energy_shift -= self._h2[i, j, l, k]

                    # elif(i in fermion_mode_array and l in fermion_mode_array
                    #      and i == j and l == k and i != l):
                    #     energy_shift += self._h2[i, j, l, k]

        # now simplify h1
        # for i in fermion_mode_array:
        #     energy_shift += h1[i, i]
        energy_shift += np.sum(np.diagonal(h1)[fermion_mode_array])
        h1_id_i, h1_id_j = np.meshgrid(mode_set_diff, mode_set_diff, indexing='ij')
        h1_new = h1[h1_id_i, h1_id_j]

        return FermionicOperator(h1_new, h2_new), energy_shift

    # def init_double_excitation_list(self, num_particles):
    #     num_orbitals = self._h1.shape[0]
    #     occupied_orbitals = np.append(np.arange(np.ceil(num_particles/2)), np.arange(
    #         num_orbitals // 2, num_orbitals // 2 + np.floor(num_particles/2))).astype(np.int32)
    #     unoccupied_orbitals = np.setdiff1d(
    #         np.arange(num_orbitals), occupied_orbitals).astype(np.int32)
    #     ret = []

    #     for i in occupied_orbitals:
    #         for j in occupied_orbitals:
    #             if i != j:
    #                 for a in unoccupied_orbitals:
    #                     for b in unoccupied_orbitals:
    #                         if a != b:
    #                             temp = (self._h2[i, a, j, b] - self._h2[i, b, j, a]) / (
    #                                 self._h1[i, i] + self._h1[j, j] - self._h1[a, a] - self._h1[b, b])
    #                             if temp != 0.0:
    #                                 ret.append([a, i, j, b, temp])
    #     return ret

    def total_particle_number(self):
        """
        TBD.
        A data_preprocess_helper fermionic operator which can be used to evaluate the number of
        particle of the given eigenstate.

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """

        size = self._h1.shape[0]
        h1 = np.eye(size, dtype=np.complex)
        h2 = np.zeros((size, size, size, size))
        return FermionicOperator(h1, h2)

    def total_magnetization(self):
        """
        TBD.

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """

        size = self._h1.shape[0]
        h1 = np.eye(size, dtype=np.complex) * 0.5
        h1[size // 2:, size // 2:] *= -1.0
        h2 = np.zeros((size, size, size, size))
        return FermionicOperator(h1, h2)

    def _S_x_squared(self):
        """

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """

        num_modes = self._h1.shape[0] // 2
        h1 = np.zeros((num_modes * 2, num_modes * 2))
        h2 = np.zeros((num_modes * 2, num_modes * 2, num_modes * 2, num_modes * 2))

        for p, q in itertools.product(range(num_modes), repeat=2):
            if p != q:
                h2[p, p + num_modes, q, q + num_modes] += 1.0
                h2[p + num_modes, p, q, q + num_modes] += 1.0
                h2[p, p + num_modes, q + num_modes, q] += 1.0
                h2[p + num_modes, p, q + num_modes, q] += 1.0
            else:
                h2[p, p + num_modes, p, p + num_modes] -= 1.0
                h2[p + num_modes, p, p + num_modes, p] -= 1.0
                h2[p, p, p + num_modes, p + num_modes] -= 1.0
                h2[p + num_modes, p + num_modes, p, p] -= 1.0

                h1[p, p] += 1.0
                h1[p + num_modes, p + num_modes] += 1.0

        h1 *= 0.25
        h2 *= 0.25
        return h1, h2

    def _S_y_squared(self):
        """

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """

        num_modes = self._h1.shape[0] // 2
        h1 = np.zeros((num_modes * 2, num_modes * 2))
        h2 = np.zeros((num_modes * 2, num_modes * 2, num_modes * 2, num_modes * 2))

        for p, q in itertools.product(range(num_modes), repeat=2):
            if p != q:
                h2[p, p + num_modes, q, q + num_modes] -= 1.0
                h2[p + num_modes, p, q, q + num_modes] += 1.0
                h2[p, p + num_modes, q + num_modes, q] += 1.0
                h2[p + num_modes, p, q + num_modes, q] -= 1.0
            else:
                h2[p, p + num_modes, p, p + num_modes] += 1.0
                h2[p + num_modes, p, p + num_modes, p] += 1.0
                h2[p, p, p + num_modes, p + num_modes] -= 1.0
                h2[p + num_modes, p + num_modes, p, p] -= 1.0

                h1[p, p] += 1.0
                h1[p + num_modes, p + num_modes] += 1.0

        h1 *= 0.25
        h2 *= 0.25
        return h1, h2

    def _S_z_squared(self):
        """

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """

        num_modes = self._h1.shape[0] // 2
        h1 = np.zeros((num_modes * 2, num_modes * 2))
        h2 = np.zeros((num_modes * 2, num_modes * 2, num_modes * 2, num_modes * 2))

        for p, q in itertools.product(range(num_modes), repeat=2):
            if p != q:
                h2[p, p, q, q] += 1.0
                h2[p + num_modes, p + num_modes, q, q] -= 1.0
                h2[p, p, q + num_modes, q + num_modes] -= 1.0
                h2[p + num_modes, p + num_modes, q + num_modes, q + num_modes] += 1.0
            else:
                h2[p, p + num_modes, p + num_modes, p] += 1.0
                h2[p + num_modes, p, p, p + num_modes] += 1.0

                h1[p, p] += 1.0
                h1[p + num_modes, p + num_modes] += 1.0

        h1 *= 0.25
        h2 *= 0.25
        return h1, h2

    def total_angular_momentum(self):
        """
        TBD.

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """

        x_h1, x_h2 = self._S_x_squared()
        y_h1, y_h2 = self._S_y_squared()
        z_h1, z_h2 = self._S_z_squared()
        h1 = x_h1 + y_h1 + z_h1
        h2 = x_h2 + y_h2 + z_h2

        return FermionicOperator(h1=h1, h2=h2)
