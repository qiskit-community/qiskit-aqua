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
Second Order Expansion feature map.
"""

import warnings
from typing import Optional, Callable, List
import numpy as np
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from .pauli_z_expansion import PauliZExpansion
from .data_mapping import self_product


class SecondOrderExpansion(PauliZExpansion):
    """DEPRECATED. Second Order Expansion feature map.

    This is a sub-class of :class:`PauliZExpansion` where *z_order* is fixed at 2.
    """

    def __init__(self,
                 feature_dimension: int,
                 depth: int = 2,
                 entangler_map: Optional[List[List[int]]] = None,
                 entanglement: str = 'full',
                 data_map_func: Callable[[np.ndarray], float] = self_product) -> None:
        """
        Args:
            feature_dimension: The number of features
            depth: The number of repeated circuits. Defaults to 2, has a minimum value of 1.
            entangler_map: Describes the connectivity of qubits, each list in the overall list
                describes [source, target]. Defaults to ``None`` where the map is created as per
                *entanglement* parameter.
                Note that the order in the list is the order of applying the two-qubit gate.
            entanglement: ('full' | 'linear'), generate the qubit connectivity by a predefined
                topology. Defaults to full which connects every qubit to each other. Linear
                connects each qubit to the next.
            data_map_func: A mapping function for data x which can be supplied to override the
                default mapping from :meth:`self_product`.
        """
        warnings.warn('The qiskit.aqua.components.feature_maps.SecondOrderExpansion object is '
                      'deprecated as of 0.7.0 and will be removed no sooner than 3 months after '
                      'the release. You should use qiskit.circuit.library.ZZFeatureMap instead.',
                      DeprecationWarning, stacklevel=2)
        validate_min('depth', depth, 1)
        validate_in_set('entanglement', entanglement, {'full', 'linear'})
        super().__init__(feature_dimension, depth, entangler_map, entanglement,
                         z_order=2, data_map_func=data_map_func)
