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

import itertools
import logging
import sys

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar

from qiskit.aqua import Operator, aqua_globals
from .qiskit_chemistry_error import QiskitChemistryError
from .bksf import bksf_mapping
from .particle_hole import particle_hole_transformation

logger = logging.getLogger(__name__)


class FermionicOperator(object):
    r"""
    A set of functions to map fermionic Hamiltonians to qubit Hamiltonians.

    References:
    - E. Wigner and P. Jordan., Über das Paulische Äguivalenzverbot, \
        Z. Phys., 47:631 (1928). \
    - S. Bravyi and A. Kitaev. Fermionic quantum computation, \
        Ann. of Phys., 298(1):210–226 (2002). \
    - A. Tranter, S. Sofia, J. Seeley, M. Kaicher, J. McClean, R. Babbush, \
        P. Coveney, F. Mintert, F. Wilhelm, and P. Love. The Bravyi–Kitaev \
        transformation: Properties and applications. Int. Journal of Quantum \
        Chemistry, 115(19):1431–1441 (2015). \
    - S. Bravyi, J. M. Gambetta, A. Mezzacapo, and K. Temme, \
        arXiv e-print arXiv:1701.08213 (2017). \
    - K. Setia, J. D. Whitfield, arXiv:1712.00446 (2017)
    """

    def __init__(self, h1, h2=None, ph_trans_shift=None):
        """Constructor.

        This class requires the integrals stored in the 'chemist' notation
            h2(i,j,k,l) --> adag_i adag_k a_m a_j
        There is another popular notation is the 'physicist' notation
            h2(i,j,k,l) --> adag_i adag_j a_k a_l
        If you are using the 'physicist' notation, you need to convert it to
        the 'chemist' notation first. E.g., h2 = numpy.einsum('ikmj->ijkm', h2)

        Args:
            h1 (numpy.ndarray): second-quantized fermionic one-body operator, a 2-D (NxN) tensor
            h2 (numpy.ndarray): second-quantized fermionic two-body operator,
                                a 4-D (NxNxNxN) tensor
            ph_trans_shift (float): energy shift caused by particle hole transformation
        """
        self._h1 = h1
        if h2 is None:
            h2 = np.zeros((h1.shape[0], h1.shape[0], h1.shape[0], h1.shape[0]), dtype=h1.dtype)
        self._h2 = h2
        self._ph_trans_shift = ph_trans_shift
        self._modes = self._h1.shape[0]
        self._map_type = None

    @property
    def modes(self):
        """Getter of modes."""
        return self._modes

    @property
    def h1(self):
        """Getter of one body integral tensor."""
        return self._h1

    @h1.setter
    def h1(self, new_h1):
        """Setter of one body integral tensor."""
        self._h1 = new_h1

    @property
    def h2(self):
        """Getter of two body integral tensor."""
        return self._h2

    @h2.setter
    def h2(self, new_h2):
        """Setter of two body integral tensor."""
        self._h2 = new_h2

    def __eq__(self, other):
        """Overload == ."""
        ret = np.all(self._h1 == other._h1)
        if not ret:
            return ret
        ret = np.all(self._h2 == other._h2)
        return ret

    def __ne__(self, other):
        """Overload != ."""
        return not self.__eq__(other)

    def transform(self, unitary_matrix):
        """Transform the one and two body term based on unitary_matrix."""
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
        temp_ret = np.zeros((num_modes, num_modes, num_modes, num_modes),
                            dtype=unitary_matrix.dtype)
        unitary_matrix_dagger = np.conjugate(unitary_matrix)

        # option 3: temp1 is a 3-D tensor, temp2 and temp3 are 2-D tensors
        for a in range(num_modes):
            temp1 = np.einsum('i,i...->...', unitary_matrix_dagger[:, a], self._h2)
            for b in range(num_modes):
                temp2 = np.einsum('j,j...->...', unitary_matrix[:, b], temp1)
                temp3 = np.einsum('kc,k...->...c', unitary_matrix_dagger, temp2)
                temp_ret[a, b, :, :] = np.einsum('ld,l...c->...cd', unitary_matrix, temp3)

        self._h2 = temp_ret

    def _jordan_wigner_mode(self, n):
        """
        Jordan_Wigner mode.

        Args:
            n (int): number of modes
        """
        a = []
        for i in range(n):
            a_z = np.asarray([1] * i + [0] + [0] * (n - i - 1), dtype=np.bool)
            a_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=np.bool)
            b_z = np.asarray([1] * i + [1] + [0] * (n - i - 1), dtype=np.bool)
            b_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=np.bool)
            a.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))
        return a

    def _parity_mode(self, n):
        """
        Parity mode.

        Args:
            n (int): number of modes
        """
        a = []
        for i in range(n):
            a_z = [0] * (i - 1) + [1] if i > 0 else []
            a_x = [0] * (i - 1) + [0] if i > 0 else []
            b_z = [0] * (i - 1) + [0] if i > 0 else []
            b_x = [0] * (i - 1) + [0] if i > 0 else []
            a_z = np.asarray(a_z + [0] + [0] * (n - i - 1), dtype=np.bool)
            a_x = np.asarray(a_x + [1] + [1] * (n - i - 1), dtype=np.bool)
            b_z = np.asarray(b_z + [1] + [0] * (n - i - 1), dtype=np.bool)
            b_x = np.asarray(b_x + [1] + [1] * (n - i - 1), dtype=np.bool)
            a.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))
        return a

    def _bravyi_kitaev_mode(self, n):
        """
        Bravyi-Kitaev mode.

        Args:
            n (int): number of modes
        """
        def parity_set(j, n):
            """Computes the parity set of the j-th orbital in n modes.

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
                indexes = np.append(indexes, parity_set(j, n / 2))
            else:
                indexes = np.append(indexes, np.append(
                    parity_set(j - n / 2, n / 2) + n / 2, n / 2 - 1))
            return indexes

        def update_set(j, n):
            """Computes the update set of the j-th orbital in n modes.

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
            """Computes the flip set of the j-th orbital in n modes.

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

            update_pauli.append(Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool)))
            parity_pauli.append(Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool)))
            remainder_pauli.append(Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool)))
            for k in range(n):
                if np.in1d(k, update_sets[j]):
                    update_pauli[j].update_x(True, k)
                if np.in1d(k, parity_sets[j]):
                    parity_pauli[j].update_z(True, k)
                if np.in1d(k, remainder_sets[j]):
                    remainder_pauli[j].update_z(True, k)

            x_j = Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool))
            x_j.update_x(True, j)
            y_j = Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool))
            y_j.update_z(True, j)
            y_j.update_x(True, j)
            a.append((update_pauli[j] * x_j * parity_pauli[j],
                      update_pauli[j] * y_j * remainder_pauli[j]))
        return a

    def mapping(self, map_type, threshold=0.00000001):
        """Map fermionic operator to qubit operator.

        Using multiprocess to speedup the mapping, the improvement can be
        observed when h2 is a non-sparse matrix.

        Args:
            map_type (str): case-insensitive mapping type.
                            "jordan_wigner", "parity", "bravyi_kitaev", "bksf"
            threshold (float): threshold for Pauli simplification

        Returns:
            Operator: create an Operator object in Paulis form.

        Raises:
            QiskitChemistryError: if the `map_type` can not be recognized.
        """
        """
        ####################################################################
        ############   DEFINING MAPPED FERMIONIC OPERATORS    ##############
        ####################################################################
        """

        self._map_type = map_type
        n = self._modes  # number of fermionic modes / qubits
        map_type = map_type.lower()
        if map_type == 'jordan_wigner':
            a = self._jordan_wigner_mode(n)
        elif map_type == 'parity':
            a = self._parity_mode(n)
        elif map_type == 'bravyi_kitaev':
            a = self._bravyi_kitaev_mode(n)
        elif map_type == 'bksf':
            return bksf_mapping(self)
        else:
            raise QiskitChemistryError('Please specify the supported modes: '
                                       'jordan_wigner, parity, bravyi_kitaev, bksf')
        """
        ####################################################################
        ############    BUILDING THE MAPPED HAMILTONIAN     ################
        ####################################################################
        """
        pauli_list = Operator(paulis=[])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Mapping one-body terms to Qubit Hamiltonian:")
            TextProgressBar(output_handler=sys.stderr)
        results = parallel_map(FermionicOperator._one_body_mapping,
                               [(self._h1[i, j], a[i], a[j])
                                for i, j in itertools.product(range(n), repeat=2) if self._h1[i, j] != 0],
                               task_args=(threshold,), num_processes=aqua_globals.num_processes)
        for result in results:
            pauli_list += result
        pauli_list.chop(threshold=threshold)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Mapping two-body terms to Qubit Hamiltonian:")
            TextProgressBar(output_handler=sys.stderr)
        results = parallel_map(FermionicOperator._two_body_mapping,
                               [(self._h2[i, j, k, m], a[i], a[j], a[k], a[m])
                                for i, j, k, m in itertools.product(range(n), repeat=4) if self._h2[i, j, k, m] != 0],
                               task_args=(threshold,), num_processes=aqua_globals.num_processes)
        for result in results:
            pauli_list += result
        pauli_list.chop(threshold=threshold)

        if self._ph_trans_shift is not None:
            pauli_term = [self._ph_trans_shift, Pauli.from_label('I' * self._modes)]
            pauli_list += Operator(paulis=[pauli_term])

        return pauli_list

    @staticmethod
    def _one_body_mapping(h1_ij_aij, threshold):
        """
        Subroutine for one body mapping.

        Args:
            h1_ij_aij (tuple): value of h1 at index (i,j), pauli at index i, pauli at index j
            threshold: (float): threshold to remove a pauli

        Returns:
            Operator: Operator for those paulis
        """
        h1_ij, a_i, a_j = h1_ij_aij
        pauli_list = []
        for alpha in range(2):
            for beta in range(2):
                pauli_prod = Pauli.sgn_prod(a_i[alpha], a_j[beta])
                coeff = h1_ij / 4 * pauli_prod[1] * np.power(-1j, alpha) * np.power(1j, beta)
                pauli_term = [coeff, pauli_prod[0]]
                if np.absolute(pauli_term[0]) > threshold:
                    pauli_list.append(pauli_term)
        return Operator(paulis=pauli_list)

    @staticmethod
    def _two_body_mapping(h2_ijkm_a_ijkm, threshold):
        """
        Subroutine for two body mapping. We use the chemists notation
        for the two-body term, h2(i,j,k,m) adag_i adag_k a_m a_j.

        Args:
            h2_ijkm_aijkm (tuple): value of h2 at index (i,j,k,m),
                                   pauli at index i, pauli at index j,
                                   pauli at index k, pauli at index m
            threshold: (float): threshold to remove a pauli

        Returns:
            Operator: Operator for those paulis
        """
        h2_ijkm, a_i, a_j, a_k, a_m = h2_ijkm_a_ijkm
        pauli_list = []
        for alpha in range(2):
            for beta in range(2):
                for gamma in range(2):
                    for delta in range(2):
                        pauli_prod_1 = Pauli.sgn_prod(a_i[alpha], a_k[beta])
                        pauli_prod_2 = Pauli.sgn_prod(pauli_prod_1[0], a_m[gamma])
                        pauli_prod_3 = Pauli.sgn_prod(pauli_prod_2[0], a_j[delta])

                        phase1 = pauli_prod_1[1] * pauli_prod_2[1] * pauli_prod_3[1]
                        phase2 = np.power(-1j, alpha + beta) * np.power(1j, gamma + delta)
                        pauli_term = [h2_ijkm / 16 * phase1 * phase2, pauli_prod_3[0]]
                        if np.absolute(pauli_term[0]) > threshold:
                            pauli_list.append(pauli_term)
        return Operator(paulis=pauli_list)

    def _convert_to_interleaved_spins(self):
        """Converting the spin order.

        From up-up-up-up-down-down-down-down
          to up-down-up-down-up-down-up-down
        """
        matrix = np.zeros((self._h1.shape), self._h1.dtype)
        n = matrix.shape[0]
        j = np.arange(n // 2)
        matrix[j, 2 * j] = 1.0
        matrix[j + n // 2, 2 * j + 1] = 1.0
        self.transform(matrix)

    def _convert_to_block_spins(self):
        """Converting the spin order.

        From up-down-up-down-up-down-up-down
          to up-up-up-up-down-down-down-down
        """
        matrix = np.zeros((self._h1.shape), self._h1.dtype)
        n = matrix.shape[0]
        j = np.arange(n // 2)
        matrix[2 * j, j] = 1.0
        matrix[2 * j + 1, n // 2 + j] = 1.0
        self.transform(matrix)

    # Modified for Open-Shell : 17.07.2019 by iso and bpa
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
            num_particles (list, int): number of particles, if it is a list, the first number is alpha
                                       and the second number is beta.
        """

        self._convert_to_interleaved_spins()
        h1, h2, energy_shift = particle_hole_transformation(self._modes, num_particles,
                                                            self._h1, self._h2)
        new_fer_op = FermionicOperator(h1=h1, h2=h2, ph_trans_shift=energy_shift)
        new_fer_op._convert_to_block_spins()
        return new_fer_op, energy_shift


    def fermion_mode_elimination(self, fermion_mode_array):
        """Eliminate modes.

        Generate a new fermionic operator with the modes in fermion_mode_array deleted

        Args:
            fermion_mode_array (list): orbital index for elimination

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        fermion_mode_array = np.sort(fermion_mode_array)
        n_modes_old = self._modes
        n_modes_new = n_modes_old - fermion_mode_array.size
        mode_set_diff = np.setdiff1d(np.arange(n_modes_old), fermion_mode_array)
        h1_id_i, h1_id_j = np.meshgrid(mode_set_diff, mode_set_diff, indexing='ij')
        h1_new = self._h1[h1_id_i, h1_id_j].copy()
        if np.count_nonzero(self._h2) > 0:
            h2_id_i, h2_id_j, h2_id_k, h2_id_l = np.meshgrid(
                mode_set_diff, mode_set_diff, mode_set_diff, mode_set_diff, indexing='ij')
            h2_new = self._h2[h2_id_i, h2_id_j, h2_id_k, h2_id_l].copy()
        else:
            h2_new = np.zeros((n_modes_new, n_modes_new, n_modes_new, n_modes_new))
        return FermionicOperator(h1_new, h2_new)

    def fermion_mode_freezing(self, fermion_mode_array):
        """Freezing modes and extracting its energy.

        Generate a fermionic operator with the modes in fermion_mode_array deleted and
        provide the shifted energy after freezing.

        Args:
            fermion_mode_array (list): orbital index for freezing

        Returns:
            FermionicOperator: Fermionic Hamiltonian
            float: energy of frozen modes
        """
        fermion_mode_array = np.sort(fermion_mode_array)
        n_modes_old = self._modes
        n_modes_new = n_modes_old - fermion_mode_array.size
        mode_set_diff = np.setdiff1d(np.arange(n_modes_old), fermion_mode_array)

        h1 = self._h1.copy()
        h2_new = np.zeros((n_modes_new, n_modes_new, n_modes_new, n_modes_new))

        energy_shift = 0.0
        if np.count_nonzero(self._h2) > 0:
            # First simplify h2 and renormalize original h1
            for _i, _j, _l, _k in itertools.product(range(n_modes_old), repeat=4):
                # Untouched terms
                h2_ijlk = self._h2[_i, _j, _l, _k]
                if h2_ijlk == 0.0:
                    continue
                if (_i in mode_set_diff and _j in mode_set_diff and
                        _l in mode_set_diff and _k in mode_set_diff):
                    h2_new[_i - np.where(fermion_mode_array < _i)[0].size,
                           _j - np.where(fermion_mode_array < _j)[0].size,
                           _l - np.where(fermion_mode_array < _l)[0].size,
                           _k - np.where(fermion_mode_array < _k)[0].size] = h2_ijlk
                else:
                    if _i in fermion_mode_array:
                        if _l not in fermion_mode_array:
                            if _i == _k and _j not in fermion_mode_array:
                                h1[_l, _j] -= h2_ijlk
                            elif _i == _j and _k not in fermion_mode_array:
                                h1[_l, _k] += h2_ijlk
                        elif _i != _l:
                            if _j in fermion_mode_array and _i == _k and _l == _j:
                                energy_shift -= h2_ijlk
                            elif _l in fermion_mode_array and _i == _j and _l == _k:
                                energy_shift += h2_ijlk
                    elif _i not in fermion_mode_array and _l in fermion_mode_array:
                        if _l == _k and _j not in fermion_mode_array:
                            h1[_i, _j] += h2_ijlk
                        elif _l == _j and _k not in fermion_mode_array:
                            h1[_i, _k] -= h2_ijlk

        # now simplify h1
        energy_shift += np.sum(np.diagonal(h1)[fermion_mode_array])
        h1_id_i, h1_id_j = np.meshgrid(mode_set_diff, mode_set_diff, indexing='ij')
        h1_new = h1[h1_id_i, h1_id_j]

        return FermionicOperator(h1_new, h2_new), energy_shift

    def total_particle_number(self):
        """
        A data_preprocess_helper fermionic operator which can be used to evaluate the number of
        particle of the given eigenstate.

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        modes = self._modes
        h1 = np.eye(modes, dtype=np.complex)
        h2 = np.zeros((modes, modes, modes, modes))
        return FermionicOperator(h1, h2)

    def total_magnetization(self):
        """
        A data_preprocess_helper fermionic operator which can be used to \
        evaluate the magnetization of the given eigenstate.

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        modes = self._modes
        h1 = np.eye(modes, dtype=np.complex) * 0.5
        h1[modes // 2:, modes // 2:] *= -1.0
        h2 = np.zeros((modes, modes, modes, modes))
        return FermionicOperator(h1, h2)

    def _s_x_squared(self):
        """

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        num_modes = self._modes
        num_modes_2 = num_modes // 2
        h1 = np.zeros((num_modes, num_modes))
        h2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

        for p, q in itertools.product(range(num_modes_2), repeat=2):
            if p != q:
                h2[p, p + num_modes_2, q, q + num_modes_2] += 1.0
                h2[p + num_modes_2, p, q, q + num_modes_2] += 1.0
                h2[p, p + num_modes_2, q + num_modes_2, q] += 1.0
                h2[p + num_modes_2, p, q + num_modes_2, q] += 1.0
            else:
                h2[p, p + num_modes_2, p, p + num_modes_2] -= 1.0
                h2[p + num_modes_2, p, p + num_modes_2, p] -= 1.0
                h2[p, p, p + num_modes_2, p + num_modes_2] -= 1.0
                h2[p + num_modes_2, p + num_modes_2, p, p] -= 1.0

                h1[p, p] += 1.0
                h1[p + num_modes_2, p + num_modes_2] += 1.0

        h1 *= 0.25
        h2 *= 0.25
        return h1, h2

    def _s_y_squared(self):
        """

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        num_modes = self._modes
        num_modes_2 = num_modes // 2
        h1 = np.zeros((num_modes, num_modes))
        h2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

        for p, q in itertools.product(range(num_modes_2), repeat=2):
            if p != q:
                h2[p, p + num_modes_2, q, q + num_modes_2] -= 1.0
                h2[p + num_modes_2, p, q, q + num_modes_2] += 1.0
                h2[p, p + num_modes_2, q + num_modes_2, q] += 1.0
                h2[p + num_modes_2, p, q + num_modes_2, q] -= 1.0
            else:
                h2[p, p + num_modes_2, p, p + num_modes_2] += 1.0
                h2[p + num_modes_2, p, p + num_modes_2, p] += 1.0
                h2[p, p, p + num_modes_2, p + num_modes_2] -= 1.0
                h2[p + num_modes_2, p + num_modes_2, p, p] -= 1.0

                h1[p, p] += 1.0
                h1[p + num_modes_2, p + num_modes_2] += 1.0

        h1 *= 0.25
        h2 *= 0.25
        return h1, h2

    def _s_z_squared(self):
        """

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        num_modes = self._modes
        num_modes_2 = num_modes // 2
        h1 = np.zeros((num_modes, num_modes))
        h2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

        for p, q in itertools.product(range(num_modes_2), repeat=2):
            if p != q:
                h2[p, p, q, q] += 1.0
                h2[p + num_modes_2, p + num_modes_2, q, q] -= 1.0
                h2[p, p, q + num_modes_2, q + num_modes_2] -= 1.0
                h2[p + num_modes_2, p + num_modes_2, q + num_modes_2, q + num_modes_2] += 1.0
            else:
                h2[p, p + num_modes_2, p + num_modes_2, p] += 1.0
                h2[p + num_modes_2, p, p, p + num_modes_2] += 1.0

                h1[p, p] += 1.0
                h1[p + num_modes_2, p + num_modes_2] += 1.0

        h1 *= 0.25
        h2 *= 0.25
        return h1, h2

    def total_angular_momentum(self):
        """Total angular momentum.

        A data_preprocess_helper fermionic operator which can be used to evaluate the total
        angular momentum of the given eigenstate.

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        x_h1, x_h2 = self._s_x_squared()
        y_h1, y_h2 = self._s_y_squared()
        z_h1, z_h2 = self._s_z_squared()
        h1 = x_h1 + y_h1 + z_h1
        h2 = x_h2 + y_h2 + z_h2

        return FermionicOperator(h1=h1, h2=h2)
