# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Bosonic Operator """

import copy
import logging
from typing import List, Tuple, Union, Optional

import numpy as np

from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator

logger = logging.getLogger(__name__)


class BosonicOperator:
    """ A set of functions to map bosonic Hamiltonians to qubit Hamiltonians.

    References:

    - *Veis Libor, et al., International Journal of Quantum Chemistry 116.18 (2016): 1328-1336.*
    - *McArdle Sam, et al., Chemical science 10.22 (2019): 5725-5735.*
    - *Ollitrault Pauline J., Chemical science 11 (2020): 6842-6855.*
    """

    def __init__(self,
                 h: List[List[Tuple[List[List[int]], float]]],
                 basis: List[int]) -> None:
        """
        The Bosonic operator in this class is written in the n-mode second quantization format
        (Eq. 10 in Ref. Ollitrault Pauline J., Chemical science 11 (2020): 6842-6855.)
        The second quantization operators act on a given modal in a given mode.
        self._degree is the truncation degree of the expansion (n).

        Args:
            h: Matrix elements for the n-body expansion. The format is as follows:
                h is a self._degree (n) dimensional array. For each degree n, h[n] contains
                the list [[indices, coeff]_0, [indices, coeff]_1, ...]
                where the indices is a n-entry list and each entry is of the
                shape [mode, modal1, modal2] which define the indices of the corresponding raising
                (mode, modal1) and lowering (mode, modal2) operators.
            basis: Is a list defining the number of modals per mode. E.g. for a 3 modes system
                with 4 modals per mode basis = [4,4,4].
        """

        self._basis = basis
        self._degree = len(h)
        self._num_modes = len(basis)
        self._h_mat = h

    def _direct_mapping(self, n: int) -> List[Tuple[Pauli, Pauli]]:
        """ Performs the transformation: a[i] = IIXIII +- iIIYIII.

        Args:
            n: number of qubits

        Returns:
            A list of pauli operators
        """
        paulis = []

        for i in range(n):
            a_z = np.asarray([0] * i + [0] + [0] * (n - i - 1), dtype=bool)
            a_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=bool)

            b_z = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=bool)
            b_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=bool)

            paulis.append((Pauli((a_z, a_x)), Pauli((b_z, b_x))))

        return paulis

    def _one_body_mapping(self, h1_ij_aij: Tuple[float, Pauli, Pauli]) -> WeightedPauliOperator:
        """ Subroutine for one body mapping.

        Args:
            h1_ij_aij: value of h1 at index (i,j), pauli at index i, pauli at index j

        Returns:
            Operator for those paulis
        """
        h1_ij, a_i, a_j = h1_ij_aij
        pauli_list = []
        for alpha in range(2):
            for beta in range(2):
                p_a = a_i[alpha].dot(a_j[beta])
                pauli_prod = p_a[:], (-1j) ** p_a.phase
                coeff = h1_ij / 4 * pauli_prod[1] * np.power(-1j, alpha) * np.power(1j, beta)
                pauli_term = [coeff, pauli_prod[0]]
                pauli_list.append(pauli_term)

        op = WeightedPauliOperator(pauli_list)

        return op

    def _extend(self, list1: List[Tuple[float, Pauli]], list2: List[Tuple[float, Pauli]]) \
            -> List[Tuple[float, Pauli]]:
        """ Concatenates the paulis for different modes together

        Args:
            list1: list of paulis for the first mode
            list2: list of paulis for the second mode

        Returns:
            The list of concatenated paulis
        """
        final_list = []
        for pauli1 in list1:
            for pauli2 in list2:
                p1c = copy.deepcopy(pauli1[1])
                p2c = copy.deepcopy(pauli2[1])
                if len(p2c) > 0:
                    p1c = p1c.insert(p1c.num_qubits, p2c)
                coeff = pauli1[0]*pauli2[0]
                final_list.append((coeff, p1c))

        return final_list

    def _combine(self, modes: List[int], paulis: dict, coeff: float) -> WeightedPauliOperator:
        """ Combines the paulis of each mode together in one WeightedPauliOperator.

        Args:
            modes: list with the indices of the modes to be combined
            paulis: dict containing the list of paulis for each mode
            coeff: coefficient multiplying the term

        Returns:
            WeightedPauliOperator acting on the modes given in argument
        """
        m = 0

        if m in modes:
            pauli_list = paulis[m]
        else:
            a_z = np.asarray([0] * self._basis[m], dtype=bool)
            a_x = np.asarray([0] * self._basis[m], dtype=bool)
            pauli_list = [(1, Pauli((a_z, a_x)))]

        for m in range(1, self._num_modes):
            if m in modes:
                new_list = paulis[m]
            else:
                a_z = np.asarray([0] * self._basis[m], dtype=bool)
                a_x = np.asarray([0] * self._basis[m], dtype=bool)
                new_list = [(1, Pauli((a_z, a_x)))]
            pauli_list = self._extend(pauli_list, new_list)

        new_pauli_list = []
        for pauli in pauli_list:
            new_pauli_list.append([coeff * pauli[0], pauli[1]])

        return WeightedPauliOperator(new_pauli_list)

    def mapping(self, qubit_mapping: str = 'direct',
                threshold: float = 1e-8) -> WeightedPauliOperator:
        """ Maps a bosonic operator into a qubit operator.

        Args:
            qubit_mapping: a string giving the type of mapping (only the 'direct' mapping is
                implemented at this point)
            threshold: threshold to chop the low contribution paulis

        Returns:
            A qubit operator

        Raises:
            ValueError: If requested mapping is not supported
        """
        qubit_op = WeightedPauliOperator([])

        pau = []
        for mode in range(self._num_modes):
            pau.append(self._direct_mapping(self._basis[mode]))

        if qubit_mapping == 'direct':

            for deg in range(self._degree):

                for element in self._h_mat[deg]:
                    paulis = {}
                    modes = []
                    coeff = element[1]
                    for i in range(deg+1):
                        m, bf1, bf2 = element[0][i]
                        modes.append(m)
                        paulis[m] = (self._one_body_mapping((1, pau[m][bf1], pau[m][bf2]))).paulis

                    qubit_op += self._combine(modes, paulis, coeff)

            qubit_op.chop(threshold)

        else:
            raise ValueError('Only the direct mapping is implemented')

        return qubit_op

    def direct_mapping_filtering_criterion(self, state: Union[List, np.ndarray], value: float,
                                           aux_values: Optional[List[float]] = None) -> bool:

        """ Filters out the states of irrelevant symmetries

        Args:
            state: the statevector
            value: the energy
            aux_values: the auxiliary energies

        Returns:
            True if the state is has one and only one modal occupied per mode meaning
            that the direct mapping symmetries are respected and False otherwise
        """
        # pylint: disable=unused-argument
        indices = np.nonzero(np.conj(state) * state > 1e-5)[0]
        count = 0
        for i in indices:
            bin_i = np.frombuffer(np.binary_repr(i, width=sum(self._basis)).encode('utf-8'),
                                  dtype='S1').astype(int)
            count = 0
            nqi = 0
            for m, _ in enumerate(self._basis):
                sub_bin = bin_i[nqi:nqi + self._basis[m]]
                occ_i = 0
                for idx_i in sub_bin:
                    occ_i += idx_i
                if occ_i != 1:
                    break
                count += 1
                nqi += self._basis[m]
        return bool(count == len(self._basis))

    def number_occupied_modals_per_mode(self, mode: int) -> 'BosonicOperator':
        """
        A bosonic operator which can be used to evaluate the number of
        occupied modals in a given mode

        Args:
            mode: the index of the mode

        Returns:
            BosonicOperator: the corresponding bosonic operator
        """
        h_mat: List[List[Tuple[List[List[int]], float]]] = [[]]
        for modal in range(self._basis[mode]):
            h_mat[0].append(([[mode, modal, modal]], 1.))

        return BosonicOperator(h_mat, self._basis)
