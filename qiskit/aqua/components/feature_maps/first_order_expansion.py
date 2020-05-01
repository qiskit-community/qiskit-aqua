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

from qiskit.aqua.components.feature_maps import PauliZExpansion, self_product


class FirstOrderExpansion(PauliZExpansion):
    """
    Mapping data with the first order expansion without entangling gates.

    Refer to https://arxiv.org/pdf/1804.11326.pdf for details.
    """

    CONFIGURATION = {
        'name': 'FirstOrderExpansion',
        'description': 'First order expansion for feature map',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'First_Order_Expansion_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, feature_dimension, depth=2, data_map_func=self_product):
        """Constructor.

        Args:
            feature_dimension (int): number of features
            depth (int): the number of repeated circuits
            data_map_func (Callable): a mapping function for data x
        """
        self.validate(locals())
        super().__init__(feature_dimension, depth, z_order=1, data_map_func=data_map_func)
