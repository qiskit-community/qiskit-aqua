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

"""The module for Aqua's first order derivatives."""

from .circuit_gradient_method import CircuitGradientMethod
from .lin_comb_gradient import LinCombGradient
from .param_shift_gradient import ParamShiftGradient
from .lin_comb_qfi import LinCombQFI
from .block_diag_qfi import BlockDiagQFI
from .diag_qfi import DiagQFI

__all__ = ['CircuitGradientMethod',
           'LinCombGradient',
           'ParamShiftGradient',
           'LinCombQFI',
           'BlockDiagQFI',
           'DiagQFI'
           ]
