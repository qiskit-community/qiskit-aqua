# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Aqua's Ansatz classes."""

from .ansatz import Ansatz
from .operator_ansatz import OperatorAnsatz
from .two_local_ansatz import TwoLocalAnsatz
from .ry import RY
from .ryrz import RYRZ
from .swaprz import SwapRZ
from .feature_maps import (PauliExpansion, FirstOrderExpansion, SecondOrderExpansion,
                           RawFeatureVector)

__all__ = [
    'Ansatz',
    'FirstOrderExpansion',
    'OperatorAnsatz',
    'PauliExpansion',
    'RawFeatureVector',
    'RY',
    'RYRZ',
    'SecondOrderExpansion',
    'SwapRZ',
    'TwoLocalAnsatz',
]
