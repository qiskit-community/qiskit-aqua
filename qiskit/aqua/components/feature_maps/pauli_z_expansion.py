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

"""The Pauli Z Expansion feature map."""

import warnings
from typing import Optional, Callable, List
import numpy as np
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from .pauli_expansion import PauliExpansion
from .data_mapping import self_product


class PauliZExpansion(PauliExpansion):
    """DEPRECATED. The Pauli Z Expansion feature map.

    This is a sub-class of the general :class:`PauliExpansion` but where the pauli string is fixed
    to only contain Z and where *paulis* is now created for the superclass as per the given
    *z_order*. So with default of 2 this creates ['Z', 'ZZ'] which also happens to be the default
    of the superclass. A *z_order* of 3 would be ['Z', 'ZZ', 'ZZZ'] and so on.
    """

    def __init__(self,
                 feature_dimension: int,
                 depth: int = 2,
                 entangler_map: Optional[List[List[int]]] = None,
                 entanglement: str = 'full',
                 z_order: int = 2,
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
            z_order: z order, has a min. value of 1. Creates *paulis* for superclass based on
                 the z order value, e.g. 3 would result in ['Z', 'ZZ', 'ZZZ'] where the paulis
                 contains strings of Z up to length of *z_order*
            data_map_func: A mapping function for data x which can be supplied to override the
                default mapping from :meth:`self_product`.
        """
        # extra warning since this class will be removed entirely
        warnings.warn('The qiskit.aqua.components.feature_maps.PauliZExpansion class is deprecated '
                      'as of 0.7.0 and will be removed no sooner than 3 months after the release. '
                      'You should use qiskit.circuit.library.PauliFeatureMap instead.',
                      DeprecationWarning, stacklevel=2)

        validate_min('depth', depth, 1)
        validate_in_set('entanglement', entanglement, {'full', 'linear'})
        validate_min('z_order', z_order, 1)
        pauli_string = []
        for i in range(1, z_order + 1):
            pauli_string.append('Z' * i)
        super().__init__(feature_dimension, depth, entangler_map, entanglement,
                         paulis=pauli_string, data_map_func=data_map_func)
