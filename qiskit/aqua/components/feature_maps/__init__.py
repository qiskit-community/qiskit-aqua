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

""" feature map packages """

from .feature_map import FeatureMap
from .data_mapping import self_product
from .pauli_expansion import PauliExpansion
from .pauli_z_expansion import PauliZExpansion
from .first_order_expansion import FirstOrderExpansion
from .second_order_expansion import SecondOrderExpansion
from .raw_feature_vector import RawFeatureVector

__all__ = ['FeatureMap',
           'self_product',
           'PauliExpansion',
           'PauliZExpansion',
           'FirstOrderExpansion',
           'SecondOrderExpansion',
           'RawFeatureVector'
           ]
