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
"""
This module contains the definition of a base class for
feature map. Several types of commonly used approaches.
"""

from typing import Callable
import numpy as np
from qiskit.aqua.utils.validation import validate_min
from .pauli_z_expansion import PauliZExpansion
from .data_mapping import self_product


class FirstOrderExpansion(PauliZExpansion):
    """
    Mapping data with the first order expansion without entangling gates.

    Refer to https://arxiv.org/pdf/1804.11326.pdf for details.
    """

    def __init__(self,
                 feature_dimension: int,
                 depth: int = 2,
                 data_map_func: Callable[[np.ndarray], float] = self_product) -> None:
        """Constructor.

        Args:
            feature_dimension: number of features
            depth: the number of repeated circuits, has a min. value of 1.
            data_map_func: a mapping function for data x
        """
        validate_min('depth', depth, 1)
        super().__init__(feature_dimension, depth, z_order=1, data_map_func=data_map_func)
