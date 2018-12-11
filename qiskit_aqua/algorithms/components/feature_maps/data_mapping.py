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

"""
This module contains the definition of data mapping function for feature map.
"""

import functools

import numpy as np


def self_product(x):
    """
    Define a function map from R^n to R.

    Args:
        x (np.ndarray): data

    Returns:
        double: the mapped value
    """
    coeff = x[0] if len(x) == 1 else \
        functools.reduce(lambda m, n: m * n, np.pi - x)
    return coeff
