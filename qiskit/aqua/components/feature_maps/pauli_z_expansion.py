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
"""
This module contains the definition of a base class for
feature map. Several types of commonly used approaches.
"""

from typing import Optional, Callable, List
import numpy as np
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from .pauli_expansion import PauliExpansion
from .data_mapping import self_product


class PauliZExpansion(PauliExpansion):
    """
    Mapping data with the second order expansion followed by entangling gates.

    Refer to https://arxiv.org/pdf/1804.11326.pdf for details.
    """

    def __init__(self,
                 feature_dimension: int,
                 depth: int = 2,
                 entangler_map: Optional[List[List[int]]] = None,
                 entanglement: str = 'full',
                 z_order: int = 2,
                 data_map_func: Callable[[np.ndarray], float] = self_product) -> None:
        """Constructor.

        Args:
            feature_dimension: number of features
            depth: the number of repeated circuits, has a min. value of 1.
            entangler_map : describe the connectivity of qubits, each list describes
                                        [source, target], or None for full entanglement.
                                        Note that the order is the list is the order of
                                        applying the two-qubit gate.
            entanglement: ['full', 'linear'], generate the qubit connectivity by predefined
                                topology
            z_order: z order, has a min. value of 1.
            data_map_func: a mapping function for data x
        """
        validate_min('depth', depth, 1)
        validate_in_set('entanglement', entanglement, {'full', 'linear'})
        validate_min('z_order', z_order, 1)
        pauli_string = []
        for i in range(1, z_order + 1):
            pauli_string.append('Z' * i)
        super().__init__(feature_dimension, depth, entangler_map, entanglement,
                         paulis=pauli_string, data_map_func=data_map_func)
