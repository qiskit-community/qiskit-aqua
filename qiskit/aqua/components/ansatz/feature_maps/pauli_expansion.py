# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This module contains the definition of a base class for
feature map. Several types of commonly used approaches.
"""

from typing import Optional, Callable, List, Union
import itertools
import logging

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli

from qiskit.aqua import AquaError
from qiskit.aqua.operators import evolution_instruction
from qiskit.aqua.utils.validation import validate_min
from qiskit.aqua.components.ansatz import Ansatz
from qiskit.aqua.utils import get_entangler_map, validate_entangler_map

from .data_mapping import self_product


logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class PauliExpansion(Ansatz):
    """
    Mapping data with the second order expansion followed by entangling gates.
    Refer to https://arxiv.org/pdf/1804.11326.pdf for details.
    """

    def __init__(self,
                 feature_dimension: int,
                 depth: int = 2,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 paulis: Optional[List[str]] = None,
                 data_map_func: Callable[[np.ndarray], float] = self_product,
                 insert_barriers: bool = False) -> None:
        """

        Args:
            feature_dimension: Number of features.
            depth: The number of repeated circuits. Defaults to 2, has a min. value of 1.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                or 'sca'), a list of integer-pairs specifying the indices of qubits
                entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                Default to 'full' entanglement.
            paulis: A list of strings for to-be-used paulis. Defaults to None.
                If None, ['Z', 'ZZ'] will be used.
            data_map_func: A mapping function for data x.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.
        """
        paulis = paulis if paulis is not None else ['Z', 'ZZ']
        validate_min('depth', depth, 1)

        super().__init__(insert_barriers=insert_barriers, overwrite_block_parameters=False)

        self._num_qubits = feature_dimension
        self._entanglement = entanglement
        self._pauli_strings = self._build_subset_paulis_string(paulis)
        self._data_map_func = data_map_func

        # define a hadamard layer for convenience
        hadamards = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            hadamards.h(i)
        hadamard_layer = hadamards.to_gate()

        # set the parameters
        x = ParameterVector('x', length=feature_dimension)

        # iterate over the layers
        for _ in range(depth):
            self.append(hadamard_layer)
            for pauli in self._pauli_strings:
                coeff = self._data_map_func(self._extract_data_for_rotation(pauli, x))
                p = Pauli.from_label(pauli)
                inst = evolution_instruction([[1, p]], coeff, 1)
                self.append(inst)

        print('after construction:', self.num_parameters, self.feature_dimension)

    @property
    def feature_dimension(self) -> int:
        """Returns the feature dimension (which is equal to the number of qubits).

        Returns:
            The feature dimension of this feature map.
        """
        return self.num_qubits

    def _build_subset_paulis_string(self, paulis):
        # fill out the paulis to the number of qubits
        temp_paulis = []
        for pauli in paulis:
            len_pauli = len(pauli)
            for possible_pauli_idx in itertools.combinations(range(self._num_qubits), len_pauli):
                string_temp = ['I'] * self._num_qubits
                for idx, _ in enumerate(possible_pauli_idx):
                    string_temp[-possible_pauli_idx[idx] - 1] = pauli[-idx - 1]
                temp_paulis.append(''.join(string_temp))
        # clean up string that can not be entangled.
        final_paulis = []
        for pauli in temp_paulis:
            where_z = np.where(np.asarray(list(pauli[::-1])) != 'I')[0]
            if len(where_z) == 1:
                final_paulis.append(pauli)
            else:
                is_valid = True
                for control, target in itertools.combinations(where_z, 2):
                    if [control, target] not in self.get_entangler_map():
                        is_valid = False
                        break
                if is_valid:
                    final_paulis.append(pauli)
                else:
                    logger.warning("Due to the limited entangler_map, %s is skipped.", pauli)

        logger.info("Pauli terms include: %s", final_paulis)
        return final_paulis

    def _extract_data_for_rotation(self, pauli, x):
        where_non_i = np.where(np.asarray(list(pauli[::-1])) != 'I')[0]
        x = np.asarray(x)
        return x[where_non_i]

    # TODO duplicate in TwoLocalAnsatz, move this somewhere else
    def get_entangler_map(self, offset: int = 0) -> List[List[int]]:
        """Return the specified entangler map, if self._entangler_map if it has been set previously.

        Args:
            offset: Some entanglements allow an offset argument, since the entangler map might
                differ per entanglement block (e.g. for 'sca' entanglement). This is the block
                index.

        Returns:
            A list of [control, target] pairs specifying entanglements, also known as entangler map.

        Raises:
            AquaError: Unsupported format of entanglement, if self._entanglement has the wrong
                format.
        """
        if isinstance(self._entanglement, str):
            return get_entangler_map(self._entanglement, self.num_qubits, offset)
        elif callable(self._entanglement):
            return validate_entangler_map(self._entanglement(offset), self.num_qubits)
        elif isinstance(self._entanglement, list):
            return validate_entangler_map(self._entanglement, self.num_qubits)
        else:
            raise AquaError('Unsupported format of entanglement!')
