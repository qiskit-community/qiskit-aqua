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

"""
Operator Combos
"""

from .op_vec import OpVec
from .op_sum import OpSum
from .op_composition import OpComposition
from .op_kron import OpKron

__all__ = ['OpVec',
           'OpSum',
           'OpKron',
           'OpComposition']
