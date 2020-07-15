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

import logging

import numpy as np
import copy
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import WeightedPauliOperator

logger = logging.getLogger(__name__)

class BosonicOperator(object):
    """
    A set of functions to map bosonic Hamiltonians to qubit Hamiltonians.

    References:
        1. Veis Libor, et al., International Journal of Quantum Chemistry 116.18 (2016): 1328-1336.
        2. McArdle Sam, et al., Chemical science 10.22 (2019): 5725-5735.
        3. Ollitrault Pauline J., Chemical science 11 (2020): 6842-6855.
    """

    def __init__(self, h, basis):
        """
        | The Bosonic operator in this class is written in the n-mode second quantization format
        | (Eq. 10 in Ref. 3)
        | The second quantization operators act on a given modal in a given mode.
        | self._degree is the truncation degree of the expansion (n).

        Args:
        h (numpy.ndarray): Matrix elements for the n-body expansion. The format is as follows:
            h is a self._degree (n) dimensional array.
            For each degree n, h[n] contains the list [[indices,coef]_0, [indices, coef]_1, ...]
            where the indices is a n-entry list and each entry is of the shape [mode, modal1, modal2]
            which define the indices of the corresponding raising (mode, modal1) and
            lowering (mode, modal2) operators.

        basis (list): Is a list defining the number of modals per mode. E.g. for a 3 modes system
            with 4 modals per mode basis = [4,4,4].
        """


        self._basis = basis
        self._degree = len(h)
        self._num_modes = len(basis)
        self._h = h

        self._num_qubits = self.count_qubits(basis)


    def count_qubits(self, basis:List):

        """

        Args:
            basis:

        Returns:

        """

        num_qubits = 0
        for i in basis:
            num_qubits+=i

        return num_qubits


    def direct_mapping(self, n):

        """
        a[i] = IIXIII +- iIIYIII
        """

        a = []

        for i in range(n):

            a_z = np.asarray([0] * i + [0] + [0] * (n - i - 1), dtype=np.bool)
            a_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=np.bool)

            b_z = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=np.bool)
            b_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=np.bool)

            a.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))

        return a


    def one_body_mapping(self, h1_ij_aij):
        """
        Subroutine for one body mapping.

        Args:
            h1_ij_aij (list): value of h1 at index (i,j), pauli at index i, pauli at index j

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
                pauli_list.append(pauli_term)

        op = WeightedPauliOperator(pauli_list)

        return op


    def extend(self, list1, list2):

        final_list = []
        for pauli1 in list1:
            for pauli2 in list2:
                p1 = copy.deepcopy(pauli1[1])
                p2 = copy.deepcopy(pauli2[1])
                p1.insert_paulis(paulis=p2)
                coef = pauli1[0]*pauli2[0]
                final_list.append([coef, p1])

        return final_list


    def combine(self, modes, paulis, coef):

        m=0
        idx = 0

        if m in modes:
            pauli_list = paulis[idx]
            idx+=1
        else:
            a_z = np.asarray([0] * self._basis[m], dtype=np.bool)
            a_x = np.asarray([0] * self._basis[m], dtype=np.bool)
            pauli_list = [[1, Pauli(a_z, a_x)]]

        for m in range(1, self._num_modes):
            if m in modes:
                new_list = paulis[idx]
                idx+=1
            else:
                a_z = np.asarray([0] * self._basis[m], dtype=np.bool)
                a_x = np.asarray([0] * self._basis[m], dtype=np.bool)
                new_list = [[1, Pauli(a_z, a_x)]]
            pauli_list = self.extend(pauli_list, new_list)

        for pauli in pauli_list:
            pauli[0] = coef*pauli[0]

        return WeightedPauliOperator(pauli_list)


    def mapping(self, qubit_mapping, threshold = 1e-7):

        qubit_op = WeightedPauliOperator([])

        a = []
        for mode in range(self._num_modes):
            a.append(self.direct_mapping(self._basis[mode]))

        if qubit_mapping == 'direct':

            for deg in range(self._degree):

                for element in self._h[deg]:
                    paulis = []
                    modes = []
                    coef = element[1]
                    for d in range(deg+1):
                        m, bf1, bf2 = element[0][d]
                        modes.append(m)
                        paulis.append((self.one_body_mapping([1, a[m][bf1], a[m][bf2]])).paulis)

                    qubit_op += self.combine(modes, paulis, coef)

            qubit_op.chop(threshold)

        else:
            raise ValueError('Only the direct mapping is implemented')

        return qubit_op






