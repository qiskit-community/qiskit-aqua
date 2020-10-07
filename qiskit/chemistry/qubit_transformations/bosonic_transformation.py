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
"""
This module implements a vibronic Hamiltonian operator, representing the
energy of the nuclei in a molecule.
"""

import copy

from typing import Tuple, List, Union
from enum import Enum
import numpy as np

from qiskit.quantum_info import Pauli

from qiskit.chemistry.drivers import BaseDriver #Should we have a bosonic driver
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.components.bosonic_basis import HarmonicBasis
from qiskit.aqua.operators.legacy import WeightedPauliOperator
from .qubit_operator_transformation import QubitOperatorTransformation

class TransformationType(Enum):
    """ TransformationType enum """
    HARMONIC = 'harmonic'

class QubitMappingType(Enum):
    """ QubitMappingType enum """
    DIRECT = 'direct'

class BosonicTransformation(QubitOperatorTransformation):
    """A vibronic Hamiltonian operator representing the energy of the nuclei in the molecule"""

    def __init__(self, qubit_mapping: QubitMappingType = QubitMappingType.DIRECT,
                 transformation_type: TransformationType = TransformationType.HARMONIC,
                 basis_size: Union[int, List[int]] = 2,
                 truncation: int = 3):
        """
        Args:
            qubit_mapping: a string giving the type of mapping (only the 'direct' mapping is
                implemented at this point)
            transformation_type: a string giving the modal basis.
                The Hamiltonian is expressed in this basis.
            basis_size: define the number of modals per mode. If the number of modals is the
                same for each mode, then only an int is required.
                However, if the number of modals differ depending on the mode basis_size should be
                a list of int, for example: [3,4] means 2 modes: first mode has 3 modals,
                second mode has 4 modals.
            truncation: where is the Hamiltonian expansion truncation (1 for having only
                              1-body terms, 2 for having on 1- and 2-body terms...)
        """

        self._qubit_mapping = qubit_mapping
        self._transformation_type = transformation_type
        self._basis_size = basis_size
        self._truncation_order = truncation

        self._num_modes = None
        self._h_mat = None

    def transform(self, driver: BaseDriver
                  ) -> Tuple[WeightedPauliOperator, List[WeightedPauliOperator]]:
        """
        Transformation to qubit operator from the driver

        Args:
            driver: BaseDriver

        Returns:
            qubit operator, auxiliary operators
        """
        watson = driver.run()
        self._num_modes = watson.num_modes

        if self._transformation_type == TransformationType.HARMONIC:
            if isinstance(self._basis_size, int):
                self._basis_size = [self._basis_size] * self._num_modes
            self._h_mat = HarmonicBasis(watson, self._basis_size, self._truncation_order).run()
        else:
            raise QiskitChemistryError('Unknown Transformation type')

        # take code from bosonic operator
        qubit_op = self.mapping(qubit_mapping = self._qubit_mapping)
        qubit_op.name = 'Bosonic Operator'

        aux_ops = []

        return qubit_op, aux_ops

    def add_context(self, result: BosonicResult):
        """TODO"""
        pass


    def _direct_mapping(self, n: int) -> List[Tuple[Pauli, Pauli]]:
        """ Performs the transformation: a[i] = IIXIII +- iIIYIII.

        Args:
            n: number of qubits

        Returns:
            A list of pauli operators
        """
        paulis = []

        for i in range(n):
            a_z = np.asarray([0] * i + [0] + [0] * (n - i - 1), dtype=np.bool)
            a_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=np.bool)

            b_z = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=np.bool)
            b_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=np.bool)

            paulis.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))

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
                pauli_prod = Pauli.sgn_prod(a_i[alpha], a_j[beta])
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
                p1c.insert_paulis(paulis=p2c)
                coeff = pauli1[0]*pauli2[0]
                final_list.append((coeff, p1c))

        return final_list

    def _combine(self, modes: List[int], paulis: dict,
                     coeff: float) -> WeightedPauliOperator:
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
            a_z = np.asarray([0] * self._basis[m], dtype=np.bool)
            a_x = np.asarray([0] * self._basis[m], dtype=np.bool)
            pauli_list = [(1, Pauli(a_z, a_x))]

        for m in range(1, self._num_modes):
            if m in modes:
                new_list = paulis[m]
            else:
                a_z = np.asarray([0] * self._basis[m], dtype=np.bool)
                a_x = np.asarray([0] * self._basis[m], dtype=np.bool)
                new_list = [(1, Pauli(a_z, a_x))]
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
                        paulis[m]=(self._one_body_mapping((1, pau[m][bf1],
                                                               pau[m][bf2]))).paulis

                    qubit_op += self._combine(modes, paulis, coeff)

            qubit_op.chop(threshold)

        else:
            raise ValueError('Only the direct mapping is implemented')

        return qubit_op

    def direct_mapping_filtering_criterion(self, state) -> bool:

        """ Filters out the states of irrelevant symmetries

                Args:
                    vecs: state to evaluate

                Returns:
                    True if the state is has one and only one modal occupied per mode meaning
                    that the direct mapping symmetries are respected and False otherwise

                """

        if isinstance(self._basis_size, int):
            self._basis_size = [self._basis_size] * self._num_modes

        indices = np.nonzero(np.conj(state) * state > 1e-5)[0]
        for i in indices:
            bin_i = np.frombuffer(np.binary_repr(i, width=sum(self._basis_size)).encode('utf-8'),
                                  dtype='S1').astype(int)
            count = 0
            nqi = 0
            for m in range(len(self._basis_size)):
                sub_bin = bin_i[nqi:nqi + self._basis_size[m]]
                occ_i = 0
                for idx_i in sub_bin:
                    occ_i += idx_i
                if occ_i != 1:
                    break
                count += 1
                nqi += self._basis_size[m]
        if count == len(self._basis_size):
            return True
        else:
            return False

