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

"""First Order Expansion feature map."""

import warnings
from typing import Callable
import numpy as np
from qiskit.aqua.utils.validation import validate_min
from .pauli_z_expansion import PauliZExpansion
from .data_mapping import self_product


class FirstOrderExpansion(PauliZExpansion):
    """DEPRECATED. First Order Expansion feature map.

    This is a sub-class of :class:`PauliZExpansion` where *z_order* is fixed at 1.
    As a result the first order expansion will be a feature map without entangling gates.
    """

    def __init__(self,
                 feature_dimension: int,
                 depth: int = 2,
                 data_map_func: Callable[[np.ndarray], float] = self_product) -> None:
        """
        Args:
            feature_dimension: The number of features
            depth: The number of repeated circuits. Defaults to 2, has a minimum value of 1.
            data_map_func: A mapping function for data x which can be supplied to override the
                default mapping from :meth:`self_product`.
        """
        warnings.warn('The qiskit.aqua.components.feature_maps.FirstOrderExpansion object is '
                      'deprecated as of 0.7.0 and will be removed no sooner than 3 months after '
                      'the release. You should use qiskit.circuit.library.ZFeatureMap instead.',
                      DeprecationWarning, stacklevel=2)

        validate_min('depth', depth, 1)
        super().__init__(feature_dimension, depth, z_order=1, data_map_func=data_map_func)
