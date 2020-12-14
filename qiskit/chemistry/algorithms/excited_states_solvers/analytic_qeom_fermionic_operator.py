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

""" Fermionic Operator """

import itertools
import logging
import sys

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar

from qiskit.aqua import aqua_globals
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.chemistry.qiskit_chemistry_error import QiskitChemistryError
from qiskit.chemistry.bksf import bksf_mapping
from qiskit.chemistry.particle_hole import particle_hole_transformation

logger = logging.getLogger(__name__)


class FermionicOperatorNBody:
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


    def __init__(self, hs, ph_trans_shift=None):
        """Constructor.

        This class requires the integrals stored in the 'chemist' notation
            h2(i,j,k,l) --> adag_i adag_k a_l a_j
        There is another popular notation is the 'physicist' notation
            h2(i,j,k,l) --> adag_i adag_j a_k a_l
        If you are using the 'physicist' notation, you need to convert it to
        the 'chemist' notation first. E.g., h2 = numpy.einsum('ikmj->ijkm', h2)

        Args:
            hs (list): array containing all N body second-quantized fermionic operator
            ph_trans_shift (float): energy shift caused by particle hole transformation
        """

        self._hs = hs
        self._ph_trans_shift = ph_trans_shift
        self._modes = 0
        for h in self._hs:
            if(h is not None): self._modes = h.shape[0]
        self._map_type = None

    @property
    def modes(self):
        """Getter of modes."""
        return self._modes

    @property
    def hs(self):  # pylint: disable=invalid-name
        """Getter of one body integral tensor."""
        return self._hs

    @hs.setter
    def hs(self, new_hs):  # pylint: disable=invalid-name
        """Setter of two body integral tensor."""
        self._hs = new_hs

    def _jordan_wigner_mode(self, n):
        """
        Jordan_Wigner mode.

        Each Fermionic Operator is mapped to 2 Pauli Operators, added together with the
        appropriate phase, i.e.:

        a_i^\\dagger = Z^i (X + iY) I^(n-i-1) = (Z^i X I^(n-i-1)) + i (Z^i Y I^(n-i-1))
        a_i = Z^i (X - iY) I^(n-i-1)

        This is implemented by creating an array of tuples, each including two operators.
        The phase between two elements in a tuple is implicitly assumed, and added calculated at the
        appropriate time (see for example _one_body_mapping).

        Args:
            n (int): number of modes
        Returns:
            list[Tuple]: Pauli
        """
        a_list = []
        for i in range(n):
            a_z = np.asarray([1] * i + [0] + [0] * (n - i - 1), dtype=np.bool)
            a_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=np.bool)
            b_z = np.asarray([1] * i + [1] + [0] * (n - i - 1), dtype=np.bool)
            b_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=np.bool)
            a_list.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))
        return a_list

    def _parity_mode(self, n):
        """
        Parity mode.

        Args:
            n (int): number of modes
        Returns:
            list[Tuple]: Pauli
        """
        a_list = []
        for i in range(n):
            a_z = [0] * (i - 1) + [1] if i > 0 else []
            a_x = [0] * (i - 1) + [0] if i > 0 else []
            b_z = [0] * (i - 1) + [0] if i > 0 else []
            b_x = [0] * (i - 1) + [0] if i > 0 else []
            a_z = np.asarray(a_z + [0] + [0] * (n - i - 1), dtype=np.bool)
            a_x = np.asarray(a_x + [1] + [1] * (n - i - 1), dtype=np.bool)
            b_z = np.asarray(b_z + [1] + [0] * (n - i - 1), dtype=np.bool)
            b_x = np.asarray(b_x + [1] + [1] * (n - i - 1), dtype=np.bool)
            a_list.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))
        return a_list

    def _bravyi_kitaev_mode(self, n):
        """
        Bravyi-Kitaev mode.

        Args:
            n (int): number of modes
         Returns:
             numpy.ndarray: Array of mode indexes
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
            elif j >= n / 2 and j < n - 1:  # pylint: disable=chained-comparison
                indexes = np.append(indexes, flip_set(j - n / 2, n / 2) + n / 2)
            else:
                indexes = np.append(np.append(indexes, flip_set(
                    j - n / 2, n / 2) + n / 2), n / 2 - 1)
            return indexes

        a_list = []
        # FIND BINARY SUPERSET SIZE
        bin_sup = 1
        # pylint: disable=comparison-with-callable
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
            a_list.append((update_pauli[j] * x_j * parity_pauli[j],
                           update_pauli[j] * y_j * remainder_pauli[j]))
        return a_list

    def mapping(self, map_type, threshold=0.00000001,idx=[None]*4):
        self._map_type = map_type
        n = self._modes  # number of fermionic modes / qubits
        map_type = map_type.lower()
        if map_type == 'jordan_wigner':
            a_list = self._jordan_wigner_mode(n)
        elif map_type == 'parity':
            a_list = self._parity_mode(n)
        elif map_type == 'bravyi_kitaev':
            a_list = self._bravyi_kitaev_mode(n)
        elif map_type == 'bksf':
            return bksf_mapping(self)
        else:
            raise QiskitChemistryError('Please specify the supported modes: '
                                       'jordan_wigner, parity, bravyi_kitaev, bksf')

        pauli_list = WeightedPauliOperator(paulis=[])

        for m,h in enumerate(self._hs):
            if(h is not None):
               if(idx[m] is None):
                  results = parallel_map(FermionicOperatorNBody._n_body_mapping,
                                         [FermionicOperatorNBody._prep_mapping(h[indexes],a_list,indexes)
                                          for indexes in list(itertools.product(range(n), repeat=len(h.shape)))
                                          if h[indexes] != 0], num_processes=aqua_globals.num_processes)
               else:
                  results = parallel_map(FermionicOperatorNBody._n_body_mapping,
                                         [FermionicOperatorNBody._prep_mapping(h[indexes],a_list,indexes)
                                          for indexes in idx[m] if np.abs(h[indexes])>threshold], num_processes=aqua_globals.num_processes)
               #print("IN MAPPING ",idx[m],[h[indexes] for indexes in idx[m] if np.abs(h[indexes])>threshold],[r.print_details() for r in results])
               for result in results:
                   pauli_list += result

        pauli_list.chop(threshold=threshold)

        if self._ph_trans_shift is not None:
            pauli_term = [self._ph_trans_shift, Pauli.from_label('I' * self._modes)]
            pauli_list += WeightedPauliOperator(paulis=[pauli_term])

        return pauli_list

    @staticmethod
    def _prep_mapping(h, a_list, indexes):

        h_a = [h]
        for i in indexes:
            h_a.append(a_list[i])

        return h_a

    @staticmethod
    def _n_body_mapping(h_a):

        h = h_a[0]
        a = []
        for i in range(0,len(h_a[1:]),2):
            a.append(h_a[1+i])
        for i in range(1,len(h_a[1:]),2)[::-1]:
            a.append(h_a[1+i])

        n = int(len(a)/2)

        a_lst = []

        for i in range(n):
            a_lst.append(WeightedPauliOperator([[1,a[i][0]]])+WeightedPauliOperator([[-1j,a[i][1]]]))

        for i in range(n):
            a_lst.append(WeightedPauliOperator([[1, a[n+i][0]]])+WeightedPauliOperator([[1j, a[n+i][1]]]))

        product = a_lst[0]

        for element in a_lst[1:]:
            product = product*element

        product = (h/(2**(n*2))) * product

        return product

    def total_particle_number(self):
        """
        A data_preprocess_helper fermionic operator which can be used to evaluate the number of
        particle of the given eigenstate.

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        modes = self._modes
        h = [np.eye(modes, dtype=np.complex)]
        return FermionicOperatorNBody(h)

    def total_magnetization(self):
        """
        A data_preprocess_helper fermionic operator which can be used to \
        evaluate the magnetization of the given eigenstate.

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        modes = self._modes
        h_1 = np.eye(modes, dtype=np.complex) * 0.5
        h_1[modes // 2:, modes // 2:] *= -1.0
        return FermionicOperatorNBody([h_1])

    def _s_x_squared(self):
        """

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        num_modes = self._modes
        num_modes_2 = num_modes // 2
        h_1 = np.zeros((num_modes, num_modes))
        h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

        for p, q in itertools.product(range(num_modes_2), repeat=2):  # pylint: disable=invalid-name
            if p != q:
                h_2[p, p + num_modes_2, q, q + num_modes_2] += 1.0
                h_2[p + num_modes_2, p, q, q + num_modes_2] += 1.0
                h_2[p, p + num_modes_2, q + num_modes_2, q] += 1.0
                h_2[p + num_modes_2, p, q + num_modes_2, q] += 1.0
            else:
                h_2[p, p + num_modes_2, p, p + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p, p + num_modes_2, p] -= 1.0
                h_2[p, p, p + num_modes_2, p + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p + num_modes_2, p, p] -= 1.0

                h_1[p, p] += 1.0
                h_1[p + num_modes_2, p + num_modes_2] += 1.0

        h_1 *= 0.25
        h_2 *= 0.25
        return h_1, h_2

    def _s_y_squared(self):
        """

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        num_modes = self._modes
        num_modes_2 = num_modes // 2
        h_1 = np.zeros((num_modes, num_modes))
        h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

        for p, q in itertools.product(range(num_modes_2), repeat=2):  # pylint: disable=invalid-name
            if p != q:
                h_2[p, p + num_modes_2, q, q + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p, q, q + num_modes_2] += 1.0
                h_2[p, p + num_modes_2, q + num_modes_2, q] += 1.0
                h_2[p + num_modes_2, p, q + num_modes_2, q] -= 1.0
            else:
                h_2[p, p + num_modes_2, p, p + num_modes_2] += 1.0
                h_2[p + num_modes_2, p, p + num_modes_2, p] += 1.0
                h_2[p, p, p + num_modes_2, p + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p + num_modes_2, p, p] -= 1.0

                h_1[p, p] += 1.0
                h_1[p + num_modes_2, p + num_modes_2] += 1.0

        h_1 *= 0.25
        h_2 *= 0.25
        return h_1, h_2

    def _s_z_squared(self):
        """

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        num_modes = self._modes
        num_modes_2 = num_modes // 2
        h_1 = np.zeros((num_modes, num_modes))
        h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

        for p, q in itertools.product(range(num_modes_2), repeat=2):  # pylint: disable=invalid-name
            if p != q:
                h_2[p, p, q, q] += 1.0
                h_2[p + num_modes_2, p + num_modes_2, q, q] -= 1.0
                h_2[p, p, q + num_modes_2, q + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p + num_modes_2,
                    q + num_modes_2, q + num_modes_2] += 1.0
            else:
                h_2[p, p + num_modes_2, p + num_modes_2, p] += 1.0
                h_2[p + num_modes_2, p, p, p + num_modes_2] += 1.0

                h_1[p, p] += 1.0
                h_1[p + num_modes_2, p + num_modes_2] += 1.0

        h_1 *= 0.25
        h_2 *= 0.25
        return h_1, h_2

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
        h_1 = x_h1 + y_h1 + z_h1
        h_2 = x_h2 + y_h2 + z_h2

        return FermionicOperatorNBody([h_1, h_2])
